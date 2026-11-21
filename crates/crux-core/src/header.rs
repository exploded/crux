//! A format-neutral keyword header. FITS cards and XISF `FITSKeyword` /
//! `Property` elements both end up here, so the rest of the crate never
//! cares which file format a frame came from.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub key: String,
    /// Cleaned value: string values are unquoted and trimmed.
    pub value: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub comment: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub cards: Vec<Card>,
}

impl Header {
    pub fn push(&mut self, key: impl Into<String>, value: impl Into<String>, comment: impl Into<String>) {
        self.cards.push(Card { key: key.into(), value: value.into(), comment: comment.into() });
    }

    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// First non-empty value for `key` (case-insensitive).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.cards
            .iter()
            .find(|c| c.key.eq_ignore_ascii_case(key) && !c.value.is_empty())
            .map(|c| c.value.as_str())
    }

    pub fn str(&self, key: &str) -> Option<String> {
        self.get(key).map(str::to_string)
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(parse_number)
    }

    pub fn i64(&self, key: &str) -> Option<i64> {
        self.f64(key).map(|v| v.round() as i64)
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            "T" | "t" | "true" | "True" => Some(true),
            "F" | "f" | "false" | "False" => Some(false),
            _ => None,
        }
    }

    /// Drop commentary cards; keeps the cache small.
    pub fn without_commentary(&self) -> Header {
        Header {
            cards: self
                .cards
                .iter()
                .filter(|c| !matches!(c.key.as_str(), "COMMENT" | "HISTORY" | "" | "END"))
                .cloned()
                .collect(),
        }
    }
}

/// Parse a FITS/XISF number. Accepts Fortran `D` exponents.
pub fn parse_number(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>()
        .ok()
        .or_else(|| t.replace(['D', 'd'], "E").parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

/// Clean a raw value string: strip surrounding quotes, undo `''` escapes, trim.
pub fn clean_value(raw: &str) -> String {
    let t = raw.trim();
    if t.len() >= 2 && t.starts_with('\'') && t.ends_with('\'') {
        t[1..t.len() - 1].replace("''", "'").trim().to_string()
    } else {
        t.to_string()
    }
}

/// Parse sexagesimal "05 47 42", "-68:02:00" or a plain number.
pub fn parse_sexagesimal(s: &str) -> Option<f64> {
    let t = s.trim();
    if let Some(v) = parse_number(t) {
        return Some(v);
    }
    let neg = t.starts_with('-');
    let parts: Vec<f64> = t
        .trim_start_matches(['-', '+'])
        .split(|c: char| c == ' ' || c == ':' || c.is_ascii_alphabetic())
        .filter(|p| !p.is_empty())
        .map(parse_number)
        .collect::<Option<Vec<_>>>()?;
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    let v = parts[0] + parts.get(1).unwrap_or(&0.0) / 60.0 + parts.get(2).unwrap_or(&0.0) / 3600.0;
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_strings() {
        assert_eq!(parse_number("1.5D3"), Some(1500.0));
        assert_eq!(parse_number(" 300.0 "), Some(300.0));
        assert_eq!(clean_value("'O''Neil  '"), "O'Neil");
        assert_eq!(clean_value("'Master Dark'"), "Master Dark");
    }

    #[test]
    fn sexagesimal() {
        assert!((parse_sexagesimal("05 47 42").unwrap() - 5.795).abs() < 1e-3);
        assert!((parse_sexagesimal("-68 02 00").unwrap() + 68.0333).abs() < 1e-3);
        assert_eq!(parse_sexagesimal("87.5"), Some(87.5));
    }
}
