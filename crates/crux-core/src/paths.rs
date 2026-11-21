//! Per-user folders and path rules that differ between Windows, Linux and
//! macOS.
//!
//! | | Windows | Linux | macOS |
//! |---|---|---|---|
//! | settings | `%APPDATA%\crux` | `~/.config/crux` | `~/Library/Application Support/crux` |
//! | local data | `%LOCALAPPDATA%\crux` | `~/.local/share/crux` | `~/Library/Application Support/crux` |

use std::path::{Path, PathBuf};

/// Settings: `crux.toml`, the GUI's remembered folder.
pub fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("crux"))
}

/// Local data: the library index when there's no cache folder.
pub fn data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("crux"))
}

/// A path as a comparison key. Windows paths are case-insensitive and take
/// either slash, so the key is lowercased with backslashes; elsewhere paths
/// are compared exactly (Linux treats `a.fits` and `A.fits` as two files).
pub fn key(path: &Path) -> String {
    #[cfg(windows)]
    {
        path.to_string_lossy().replace('/', "\\").to_lowercase()
    }
    #[cfg(not(windows))]
    {
        path.to_string_lossy().into_owned()
    }
}

/// The error a rename or hard link gives when the two paths are on
/// different volumes: Windows ERROR_NOT_SAME_DEVICE (17), Unix EXDEV (18).
/// (17 is EEXIST on Unix, so the codes can't be shared.)
pub fn cross_device(e: &std::io::Error) -> bool {
    #[cfg(windows)]
    const CODE: i32 = 17;
    #[cfg(not(windows))]
    const CODE: i32 = 18;
    e.raw_os_error() == Some(CODE)
}

/// Write a whole file so a reader sees the old contents or the new, never
/// half of one: write `<name>.tmp` beside it, then rename over it (rename
/// replaces an existing file on Windows and Unix alike).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Is this file an online-only cloud placeholder (OneDrive Files On-Demand,
/// Dropbox, iCloud for Windows)? Opening one makes Windows download it, so
/// crux never reads them. Only Windows marks placeholders this way.
pub fn cloud_only(meta: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const OFFLINE: u32 = 0x1000;
        const RECALL_ON_OPEN: u32 = 0x4_0000;
        const RECALL_ON_DATA_ACCESS: u32 = 0x40_0000;
        meta.file_attributes() & (OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS) != 0
    }
    #[cfg(not(windows))]
    {
        let _ = meta;
        false
    }
}

/// Too broad to look through for imaging data: a drive or filesystem root,
/// the home folder or anything above it (`C:\`, `C:\Users`). Opening
/// `C:\PixInsight` once made crux scan every folder in `C:\` as "the folders
/// beside it", OneDrive included (2026-11-20).
pub fn too_broad(dir: &Path) -> bool {
    dir.parent().is_none() || dirs::home_dir().is_some_and(|h| PathBuf::from(key(&h)).starts_with(PathBuf::from(key(dir))))
}

/// Line ending for text files people open in an editor.
pub const EOL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_broad_roots_and_home() {
        let home = dirs::home_dir().unwrap();
        assert!(too_broad(&home));
        assert!(too_broad(home.parent().unwrap()));
        assert!(too_broad(Path::new(if cfg!(windows) { "C:\\" } else { "/" })));
        assert!(!too_broad(&home.join("Pictures")));
        #[cfg(windows)]
        {
            assert!(!too_broad(Path::new("C:\\PixInsight")));
            assert!(!too_broad(Path::new("C:\\crux-work\\testdata")));
        }
    }
}
