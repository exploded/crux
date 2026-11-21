//! Zero-copy exports: a selection as NTFS hard links in one folder (for
//! WBPP or Siril), or a file list when the files are on another volume.
//! Never overwrites; never touches the originals.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub linked: usize,
    pub bytes: u64,
    /// Names already in the export folder (left alone).
    pub existing: usize,
    /// Written when hard links aren't possible (another volume).
    pub list: Option<PathBuf>,
}

/// Hard-link `files` into `dest` (created if needed). If the first link
/// fails because `dest` is on another volume, writes `dest\files.txt`
/// listing the paths instead.
pub fn hard_links(files: &[PathBuf], dest: &Path) -> Result<Outcome> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut out = Outcome::default();
    for f in files {
        let Some(name) = f.file_name() else { continue };
        let to = dest.join(name);
        if to.exists() {
            out.existing += 1;
            continue;
        }
        match std::fs::hard_link(f, &to) {
            Ok(()) => {
                out.linked += 1;
                out.bytes += std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            }
            Err(e) if crate::paths::cross_device(&e) && out.linked == 0 => {
                let list = dest.join("files.txt");
                let body: String = files.iter().map(|p| format!("{}{}", p.display(), crate::paths::EOL)).collect();
                std::fs::write(&list, body).with_context(|| format!("write {}", list.display()))?;
                out.list = Some(list);
                return Ok(out);
            }
            Err(e) => return Err(e).with_context(|| format!("link {} to {}", f.display(), to.display())),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_without_copying() {
        let d = std::env::temp_dir().join(format!("crux-export-{}", std::process::id()));
        let src = d.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let a = src.join("a.fits");
        std::fs::write(&a, b"abc").unwrap();
        let out = hard_links(&[a.clone()], &d.join("out")).unwrap();
        assert_eq!((out.linked, out.bytes, out.existing), (1, 3, 0));
        assert_eq!(std::fs::read(d.join("out").join("a.fits")).unwrap(), b"abc");
        let again = hard_links(&[a], &d.join("out")).unwrap();
        assert_eq!((again.linked, again.existing), (0, 1));
        std::fs::remove_dir_all(&d).ok();
    }
}
