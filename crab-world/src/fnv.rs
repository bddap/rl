//! FNV-1a/64 — the one build-stable, allocation-free hash the determinism guards share.
//!
//! Every cross-peer agreement and desync check in the game folds bytes with FNV-1a/64:
//! the GCR lockstep state hash (`net::sim::Sim::state_hash`), the crab physics
//! digest ([`crate::bot::physics_digest`]), the policy-weights digest
//! ([`crate::play::policy`]), and the membership roster token (`net::membership`).
//! They MUST agree byte-for-byte — `Sim::state_hash` folds the physics digest in directly —
//! so the offset basis, the prime, and the xor-then-multiply loop live here ONCE. A constant
//! or step that drifted between copies would silently desync honest peers (rl#96, rl#102).
//!
//! Unlike `std::hash::DefaultHasher`, the algorithm and seed are fixed in-code, so two peers
//! built from the same binary hash identical bytes to an identical value across processes and
//! machines — the property every guard above needs and `DefaultHasher` explicitly does not
//! guarantee.

/// FNV-1a/64 offset basis — the rolling hash's start value.
pub const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a/64 prime — the per-byte multiplier.
pub const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Streaming FNV-1a/64 hasher: seed with [`Fnv::new`], fold bytes with [`Fnv::write`] in any
/// number of calls, read the digest with [`Fnv::finish`]. `Copy`, allocation-free, no internal
/// buffering — `write` consumes each byte immediately, so chaining many small writes is
/// identical to one write over the concatenation.
#[derive(Clone, Copy)]
#[must_use]
pub struct Fnv(u64);

impl Fnv {
    pub fn new() -> Self {
        Self(OFFSET_BASIS)
    }

    /// Resume folding from a previously [`finish`](Fnv::finish)ed digest, so a digest can be
    /// extended by further bytes. Folding into `Fnv::resume(a)` the little-endian bytes of `b`
    /// is the agreement-token construction in `net::membership`.
    pub fn resume(state: u64) -> Self {
        Self(state)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    #[must_use]
    pub fn finish(self) -> u64 {
        self.0
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot FNV-1a/64 over a byte slice, seeded from [`OFFSET_BASIS`] — the build-stable hash
/// the GCR guards use to compare BYTE blobs across peers (a checkpoint's weights, the crab-model
/// asset, a roster). Two same-binary peers hash identical bytes to an identical value.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical FNV-1a/64 test vectors (the published reference values). They pin the
    // implementation to the standard algorithm AND to the exact constants every deduped copy
    // used, so a future edit that changed either constant fails here and at the determinism
    // tests rather than silently desyncing peers.
    #[test]
    fn known_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let mut h = Fnv::new();
        h.write(b"foo");
        h.write(b"bar");
        assert_eq!(h.finish(), fnv1a(b"foobar"));
    }

    #[test]
    fn resume_continues_the_fold() {
        // Resuming from a finished digest and folding more bytes equals hashing the whole
        // sequence in one pass — the property agreement_token relies on.
        let a = fnv1a(b"foo");
        let mut h = Fnv::resume(a);
        h.write(b"bar");
        assert_eq!(h.finish(), fnv1a(b"foobar"));
    }
}
