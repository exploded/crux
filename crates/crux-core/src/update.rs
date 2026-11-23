//! Is there a newer crux? Asks GitHub for the latest published release of
//! `exploded/crux` (drafts and pre-releases aren't listed), at most once a
//! day: the answer is kept in `update.json` in the settings folder.
//! Nothing about the user or their images is sent, only crux's version in
//! the User-Agent.
//!
//! The request needs the `update` feature (ureq with rustls, whose `ring`
//! compiles C, so `build.bat platforms` builds without it). Without it,
//! [`status`] says the check isn't in this build. `[updates] check =
//! false` in `crux.toml`, or `CRUX_UPDATE_CHECK=off`, turns it off.
//! `CRUX_UPDATE_PRETEND_VERSION=0.0.1` makes crux think it's that version
//! (to see the update notice without a new release).

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// The releases page answers this; crux reads `tag_name`, `html_url` and
/// `published_at`.
pub const LATEST_URL: &str = "https://api.github.com/repos/exploded/crux/releases/latest";
/// Where to send people when they want to look themselves.
pub const RELEASES_PAGE: &str = "https://github.com/exploded/crux/releases";
/// How long an answer is reused before GitHub is asked again.
const MAX_AGE_HOURS: i64 = 24;

/// A published release.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Release {
    /// Without the tag's leading `v`: "0.2.0".
    pub version: String,
    /// The release's page on GitHub, with its downloads.
    pub url: String,
}

/// What the check found.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// Not asked yet this session.
    NotChecked,
    /// Turned off in Settings or by `CRUX_UPDATE_CHECK=off`.
    Off,
    /// Built without the `update` feature.
    NotInBuild,
    Checking,
    /// This is the latest release (or newer). `checked` is when GitHub
    /// was asked, local time.
    UpToDate { latest: String, checked: chrono::DateTime<chrono::Local> },
    /// A newer release is out.
    Newer { release: Release, checked: chrono::DateTime<chrono::Local> },
    /// GitHub couldn't be reached or gave an unexpected answer.
    Failed(String),
}

static STATUS: Mutex<Status> = Mutex::new(Status::NotChecked);

/// The version this build calls itself (`CRUX_UPDATE_PRETEND_VERSION`
/// wins, for testing the notice).
pub fn current() -> String {
    std::env::var("CRUX_UPDATE_PRETEND_VERSION").ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// Whether this build can ask GitHub at all.
pub fn in_build() -> bool {
    cfg!(feature = "update")
}

/// The latest answer, for the top bar and Settings.
pub fn status() -> Status {
    STATUS.lock().map(|s| s.clone()).unwrap_or(Status::NotChecked)
}

fn set(s: Status) {
    if let Ok(mut g) = STATUS.lock() {
        *g = s;
    }
}

/// Settings turned the check off: forget any notice.
pub fn turn_off() {
    set(Status::Off);
}

fn env_off() -> bool {
    std::env::var("CRUX_UPDATE_CHECK").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"))
}

/// Check in the background (once a day unless `force`) and call `done`
/// when the answer is in. `enabled` is `[updates] check`; a forced check
/// (Settings' Check now) runs even when that's off.
pub fn start(enabled: bool, force: bool, done: impl FnOnce() + Send + 'static) {
    if !in_build() {
        set(Status::NotInBuild);
        return;
    }
    if (!enabled || env_off()) && !force {
        set(Status::Off);
        return;
    }
    if status() == Status::Checking {
        return;
    }
    set(Status::Checking);
    std::thread::spawn(move || {
        set(check(force));
        done();
    });
}

/// Ask GitHub, or reuse an answer under a day old, and compare.
fn check(force: bool) -> Status {
    let file = crate::paths::config_dir().map(|d| d.join("update.json"));
    check_in(file.as_deref(), force, chrono::Utc::now(), fetch)
}

fn check_in(file: Option<&std::path::Path>, force: bool, now: chrono::DateTime<chrono::Utc>, fetch: impl FnOnce() -> anyhow::Result<Release>) -> Status {
    let saved = file.and_then(Saved::read);
    let fresh = saved.filter(|s| !force && (0..MAX_AGE_HOURS).contains(&now.signed_duration_since(s.checked).num_hours()));
    let (release, checked) = match fresh {
        Some(s) => (s.release, s.checked),
        None => match fetch() {
            Ok(r) => {
                if let Some(f) = file {
                    let s = Saved { checked: now, release: r.clone() };
                    if let Some(d) = f.parent() {
                        let _ = std::fs::create_dir_all(d);
                    }
                    if let Ok(b) = serde_json::to_vec_pretty(&s) {
                        let _ = crate::paths::write_atomic(f, &b);
                    }
                }
                (r, now)
            }
            Err(e) => return Status::Failed(format!("{e:#}")),
        },
    };
    let checked = checked.with_timezone(&chrono::Local);
    if newer(&current(), &release.version) {
        Status::Newer { release, checked }
    } else {
        Status::UpToDate { latest: release.version, checked }
    }
}

/// `update.json`: the last answer and when it came.
#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    checked: chrono::DateTime<chrono::Utc>,
    #[serde(flatten)]
    release: Release,
}

impl Saved {
    fn read(p: &std::path::Path) -> Option<Saved> {
        serde_json::from_slice(&std::fs::read(p).ok()?).ok()
    }
}

/// Is `latest` a later version than `current`? Versions that don't parse
/// never count as newer.
pub fn newer(current: &str, latest: &str) -> bool {
    match (parse(current), parse(latest)) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

fn parse(v: &str) -> Option<semver::Version> {
    let v = v.trim();
    semver::Version::parse(v.strip_prefix(['v', 'V']).unwrap_or(v)).ok()
}

/// Read GitHub's answer for the latest release.
pub fn parse_release(json: &str) -> anyhow::Result<Release> {
    #[derive(Deserialize)]
    struct Gh {
        tag_name: String,
        html_url: String,
    }
    let g: Gh = serde_json::from_str(json)?;
    let tag = g.tag_name.trim();
    let version = tag.strip_prefix(['v', 'V']).unwrap_or(tag).to_string();
    anyhow::ensure!(parse(&version).is_some(), "the latest release's tag isn't a version: {tag}");
    Ok(Release { version, url: g.html_url })
}

#[cfg(feature = "update")]
fn fetch() -> anyhow::Result<Release> {
    use std::time::Duration;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .user_agent(format!("crux/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut resp = agent.get(LATEST_URL).header("Accept", "application/vnd.github+json").call().map_err(|e| match e {
        ureq::Error::StatusCode(404) => anyhow::anyhow!("no release has been published yet"),
        ureq::Error::StatusCode(403 | 429) => anyhow::anyhow!("GitHub is limiting requests from this address; try again later"),
        e => anyhow::anyhow!("couldn't reach GitHub: {e}"),
    })?;
    let body = resp.body_mut().with_config().limit(1 << 20).read_to_string()?;
    parse_release(&body)
}

#[cfg(not(feature = "update"))]
fn fetch() -> anyhow::Result<Release> {
    anyhow::bail!("this build has no update check")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_as_versions() {
        assert!(newer("0.1.0", "0.2.0"));
        assert!(newer("0.1.0", "v0.1.1"));
        assert!(newer("0.9.0", "0.10.0"));
        assert!(newer("0.2.0-rc1", "0.2.0"));
        assert!(!newer("0.2.0", "0.2.0-rc1"));
        assert!(!newer("0.2.0", "0.2.0"));
        assert!(!newer("0.2.0", "0.1.9"));
        assert!(!newer("0.1.0", "nightly"));
    }

    #[test]
    fn reads_githubs_answer() {
        let json = r#"{"url":"https://api.github.com/repos/exploded/crux/releases/1","html_url":"https://github.com/exploded/crux/releases/tag/v0.2.0","tag_name":"v0.2.0","name":"crux v0.2.0","draft":false,"prerelease":false,"published_at":"2026-12-01T10:00:00Z","assets":[]}"#;
        let r = parse_release(json).unwrap();
        assert_eq!(r.version, "0.2.0");
        assert_eq!(r.url, "https://github.com/exploded/crux/releases/tag/v0.2.0");
        assert!(parse_release(r#"{"tag_name":"latest","html_url":"x"}"#).is_err());
        assert!(parse_release(r#"{"message":"Not Found"}"#).is_err());
    }

    #[test]
    fn asks_github_at_most_once_a_day() {
        let dir = std::env::temp_dir().join(format!("crux-update-test-{}", std::process::id()));
        let f = dir.join("update.json");
        let _ = std::fs::remove_dir_all(&dir);
        let rel = |v: &str| Release { version: v.into(), url: format!("https://github.com/exploded/crux/releases/tag/v{v}") };
        let t0 = chrono::Utc::now();
        let hours = |h: i64| t0 + chrono::Duration::hours(h);

        // Nothing saved: asks, and saves the answer.
        let s = check_in(Some(&f), false, t0, || Ok(rel("999.0.0")));
        assert!(matches!(s, Status::Newer { ref release, .. } if release.version == "999.0.0"), "{s:?}");
        assert!(f.is_file());
        // Within a day: the saved answer, no request.
        let s = check_in(Some(&f), false, hours(23), || panic!("asked again within a day"));
        assert!(matches!(s, Status::Newer { .. }), "{s:?}");
        // Forced: asks.
        let s = check_in(Some(&f), true, hours(1), || Ok(rel("0.0.1")));
        assert!(matches!(s, Status::UpToDate { ref latest, .. } if latest == "0.0.1"), "{s:?}");
        // A day later: asks; a failure says so and keeps the old answer.
        let s = check_in(Some(&f), false, hours(26), || anyhow::bail!("offline"));
        assert_eq!(s, Status::Failed("offline".into()));
        assert_eq!(Saved::read(&f).unwrap().release.version, "0.0.1");
        // A saved time in the future (clock changed) doesn't stop checks.
        let s = check_in(Some(&f), false, hours(-5), || Ok(rel("999.1.0")));
        assert!(matches!(s, Status::Newer { ref release, .. } if release.version == "999.1.0"), "{s:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Asks the real GitHub API. `cargo test -p crux-core update -- --ignored`.
    #[cfg(feature = "update")]
    #[test]
    #[ignore]
    fn asks_github() {
        let r = fetch().unwrap();
        println!("latest release {} at {}", r.version, r.url);
        assert!(parse(&r.version).is_some());
    }
}
