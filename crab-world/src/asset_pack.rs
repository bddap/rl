//! The baked-asset container (rl#411 stage 6): ONE opaque blob holding the full
//! asset tree, so a hosted web build serves compiled artifacts only — never the raw
//! asset files, which stay in the private assets repo (owner directive 2026-08-25).
//! The writer runs in the packager (`pack-assets`), the parser in the web entry;
//! sharing one module makes format drift impossible.
//!
//! Format (all integers LE):
//! `RLPACK1\n` magic, u32 entry count, then per entry: u32 path length, the
//! asset-tree-relative UTF-8 path, u64 raw length, u64 deflate length, the
//! raw-deflate bytes. Entries are sorted by path — the writer's output is a pure
//! function of the tree, so identical trees pack byte-identically.

use std::io::{Error, ErrorKind, Read, Result};

const MAGIC: &[u8; 8] = b"RLPACK1\n";

/// Serialize `entries` (asset-tree-relative path, raw bytes) into a pack blob.
/// Deflate is worth real bytes here: the WAV ambience beds and checkpoint tensors
/// both shrink substantially, and the pack is fetched whole on every page load.
pub fn write_pack(mut entries: Vec<(String, Vec<u8>)>) -> Result<Vec<u8>> {
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(w) = entries.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("duplicate pack entry: {}", w[0].0),
        ));
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    let count = u32::try_from(entries.len())
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "too many pack entries"))?;
    out.extend_from_slice(&count.to_le_bytes());
    for (path, raw) in entries {
        let mut enc =
            flate2::read::DeflateEncoder::new(raw.as_slice(), flate2::Compression::default());
        let mut comp = Vec::new();
        enc.read_to_end(&mut comp)?;
        let path_len = u32::try_from(path.len())
            .map_err(|_| Error::new(ErrorKind::InvalidInput, format!("path too long: {path}")))?;
        out.extend_from_slice(&path_len.to_le_bytes());
        out.extend_from_slice(path.as_bytes());
        out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        out.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        out.extend_from_slice(&comp);
    }
    Ok(out)
}

/// Parse a pack blob back into (path, raw bytes) entries. Any structural fault is a
/// hard error — a truncated or corrupt pack must refuse to boot the game (rl#375),
/// never yield a partial tree.
pub fn read_pack(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    fn bad(what: &str) -> Error {
        Error::new(ErrorKind::InvalidData, format!("asset pack: {what}"))
    }
    fn take<'a>(rest: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8]> {
        if rest.len() < n {
            return Err(bad(&format!("truncated reading {what}")));
        }
        let (head, tail) = rest.split_at(n);
        *rest = tail;
        Ok(head)
    }
    let mut rest = bytes
        .strip_prefix(MAGIC.as_slice())
        .ok_or_else(|| bad("bad magic (not an RLPACK1 blob)"))?;
    let count = u32::from_le_bytes(take(&mut rest, 4, "entry count")?.try_into().unwrap());
    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let path_len =
            u32::from_le_bytes(take(&mut rest, 4, "path length")?.try_into().unwrap()) as usize;
        let path = std::str::from_utf8(take(&mut rest, path_len, "path")?)
            .map_err(|_| bad("path not UTF-8"))?
            .to_string();
        // Checked u64→usize: on wasm32 a plain `as usize` truncates, and a corrupt
        // header claiming `k + 2^32` would slip past the inflated-length check below.
        let raw_len = usize::try_from(u64::from_le_bytes(
            take(&mut rest, 8, "raw length")?.try_into().unwrap(),
        ))
        .map_err(|_| bad("raw length overflows this platform"))?;
        let comp_len = usize::try_from(u64::from_le_bytes(
            take(&mut rest, 8, "deflate length")?.try_into().unwrap(),
        ))
        .map_err(|_| bad("deflate length overflows this platform"))?;
        let comp = take(&mut rest, comp_len, &format!("data of {path}"))?;
        let mut raw = Vec::with_capacity(raw_len);
        flate2::read::DeflateDecoder::new(comp).read_to_end(&mut raw)?;
        if raw.len() != raw_len {
            return Err(bad(&format!(
                "{path}: inflated {} bytes, header says {raw_len}",
                raw.len()
            )));
        }
        entries.push((path, raw));
    }
    if !rest.is_empty() {
        return Err(bad("trailing bytes after last entry"));
    }
    Ok(entries)
}

/// Pack every regular file under `root`, paths relative to it — the packager's one
/// walk, here so the format and the tree traversal can't drift apart.
pub fn pack_dir(root: &std::path::Path) -> Result<Vec<u8>> {
    let mut entries = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| Error::new(ErrorKind::InvalidData, e))?
                    .to_str()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("non-UTF-8 asset path: {}", path.display()),
                        )
                    })?
                    .to_string();
                entries.push((rel, std::fs::read(&path)?));
            }
        }
    }
    write_pack(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let entries = vec![
            ("weights/brain.bin".to_string(), vec![7u8; 4096]),
            ("sally.glb".to_string(), b"glTF-binary-ish".to_vec()),
            ("empty.txt".to_string(), Vec::new()),
        ];
        let pack = write_pack(entries.clone()).unwrap();
        let mut sorted = entries;
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(read_pack(&pack).unwrap(), sorted);
    }

    #[test]
    fn deterministic_and_ordered() {
        let a = write_pack(vec![("b".into(), vec![1, 2, 3]), ("a".into(), vec![4, 5])]).unwrap();
        let b = write_pack(vec![("a".into(), vec![4, 5]), ("b".into(), vec![1, 2, 3])]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn duplicate_paths_refused() {
        assert!(write_pack(vec![("a".into(), vec![1]), ("a".into(), vec![2])]).is_err());
    }

    #[test]
    fn corrupt_pack_refuses() {
        let pack = write_pack(vec![("x".into(), vec![9; 100])]).unwrap();
        assert!(read_pack(&pack[..pack.len() - 1]).is_err(), "truncated");
        assert!(read_pack(b"NOTAPACK").is_err(), "bad magic");
        let mut trailing = pack;
        trailing.push(0);
        assert!(read_pack(&trailing).is_err(), "trailing bytes");
    }
}
