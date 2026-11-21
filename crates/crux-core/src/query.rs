//! The smart-view query language, shared by the GUI's query bar and
//! `crux find`. Space-separated terms, all of which must match:
//!
//! - `key:value` — `panel:12`, `filter:H`, `night:2026-11` (prefix),
//!   `grade:reject`, `type:flat` (light, flat, dark, bias, master,
//!   processed), `pier:east`, `target:lmc` and `folder:bad` (contains),
//!   `reason:cloud` (main reject reason).
//! - comparisons — `t<0.8`, `fwhm<=4.5`, `ecc>0.6`, `stars>3000`,
//!   `sky<0.2`, `alt>30`, `exp=300`.
//! - anything else — part of the path (case-insensitive).
//!
//! Comparisons on a measure a file doesn't have (headers only) don't match.

use crate::index::IndexFile;
use crate::{FrameType, Role};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Key(String, String),
    Cmp(String, Op, f64),
    Text(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Query {
    pub terms: Vec<Term>,
    /// Terms that look like filters but aren't (e.g. `colour:red`).
    pub unknown: Vec<String>,
}

const KEYS: [&str; 9] = ["panel", "filter", "night", "grade", "type", "pier", "target", "folder", "reason"];
const FIELDS: [&str; 7] = ["t", "fwhm", "ecc", "stars", "sky", "alt", "exp"];

pub fn parse(q: &str) -> Query {
    let mut out = Query::default();
    for tok in q.split_whitespace() {
        let low = tok.to_lowercase();
        if let Some((k, v)) = low.split_once(':') {
            if KEYS.contains(&k) && !v.is_empty() {
                out.terms.push(Term::Key(k.to_string(), v.to_string()));
            } else {
                out.unknown.push(tok.to_string());
            }
            continue;
        }
        if let Some(t) = comparison(&low) {
            out.terms.push(t);
            continue;
        }
        out.terms.push(Term::Text(low));
    }
    out
}

fn comparison(s: &str) -> Option<Term> {
    for (sym, op) in [("<=", Op::Le), (">=", Op::Ge), ("<", Op::Lt), (">", Op::Gt), ("=", Op::Eq)] {
        if let Some((f, v)) = s.split_once(sym) {
            if FIELDS.contains(&f) {
                return v.parse().ok().map(|n| Term::Cmp(f.to_string(), op, n));
            }
        }
    }
    None
}

/// The `type:` value of a file.
pub fn type_name(f: &IndexFile) -> &'static str {
    match (f.role, f.kind) {
        (Role::Master, _) => "master",
        (Role::Processed, _) => "processed",
        (_, FrameType::Light) => "light",
        (_, FrameType::Flat) => "flat",
        (_, FrameType::Dark) | (_, FrameType::DarkFlat) => "dark",
        (_, FrameType::Bias) => "bias",
        _ => "unknown",
    }
}

impl Query {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn matches(&self, f: &IndexFile) -> bool {
        self.terms.iter().all(|t| term_matches(t, f))
    }
}

fn term_matches(t: &Term, f: &IndexFile) -> bool {
    match t {
        Term::Key(k, v) => match k.as_str() {
            "panel" => f.panel.is_some_and(|p| p.to_string() == v.trim_start_matches('0') || (v == "0" && p == 0)),
            "filter" => f.filter.as_deref().is_some_and(|x| x.eq_ignore_ascii_case(v)),
            "night" => f.night.is_some_and(|n| n.to_string().starts_with(v.as_str())),
            "grade" => f.grade.is_some_and(|g| g.name() == v),
            "type" => type_name(f) == v || (v == "light" && type_name(f) == "light"),
            "pier" => f.pier.as_deref().is_some_and(|p| p.to_lowercase().contains(v.as_str())),
            "target" => f.target.as_deref().is_some_and(|x| x.to_lowercase().contains(v.as_str())),
            "folder" => f.path.parent().is_some_and(|p| p.to_string_lossy().to_lowercase().contains(v.as_str())),
            "reason" => f.reason.as_deref().is_some_and(|r| r.to_lowercase().contains(v.as_str())),
            _ => false,
        },
        Term::Cmp(field, op, n) => {
            let v = match field.as_str() {
                "t" => f.t,
                "fwhm" => f.fwhm,
                "ecc" => f.ecc,
                "stars" => f.stars.map(f64::from),
                "sky" => f.sky,
                "alt" => f.alt,
                "exp" => f.exptime,
                _ => None,
            };
            v.is_some_and(|v| match op {
                Op::Lt => v < *n,
                Op::Le => v <= *n,
                Op::Gt => v > *n,
                Op::Ge => v >= *n,
                Op::Eq => (v - n).abs() < 1e-6 * n.abs().max(1.0),
            })
        }
        Term::Text(s) => f.path.to_string_lossy().to_lowercase().contains(s.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grade::Grade;

    fn light() -> IndexFile {
        IndexFile {
            path: ["PixInsight", "LMC", "LIGHT", "PANEL_12", "2026-11-11_21-29-21_H_-10.00_300.00s_0005.fits"].iter().collect(),
            kind: FrameType::Light,
            role: Role::Raw,
            target: Some("LMC".into()),
            panel: Some(12),
            filter: Some("H".into()),
            night: "2026-11-11".parse().ok(),
            exptime: Some(300.0),
            grade: Some(Grade::Warn),
            reason: Some("tracking".into()),
            t: Some(0.96),
            fwhm: Some(4.2),
            pier: Some("West".into()),
            ..Default::default()
        }
    }

    #[test]
    fn filters() {
        let f = light();
        for q in ["panel:12", "filter:h", "night:2026-11", "grade:warn", "type:light", "pier:west", "target:lmc", "folder:panel_12", "t>0.9", "fwhm<4.5", "exp=300", "21-29", "panel:12 fwhm<=4.2 grade:warn", ""] {
            assert!(parse(q).matches(&f), "{q}");
        }
        for q in ["panel:13", "filter:o", "night:2026-10", "grade:good", "type:flat", "t<0.5", "ecc>0.5", "M42"] {
            assert!(!parse(q).matches(&f), "{q}");
        }
        let p = parse("colour:red t<0.8");
        assert_eq!(p.unknown, vec!["colour:red".to_string()]);
        assert_eq!(p.terms, vec![Term::Cmp("t".into(), Op::Lt, 0.8)]);
    }
}
