pub const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

pub const PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
#[must_use]
pub struct Fnv(u64);

impl Fnv {
    pub fn new() -> Self {
        Self(OFFSET_BASIS)
    }

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

#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let a = fnv1a(b"foo");
        let mut h = Fnv::resume(a);
        h.write(b"bar");
        assert_eq!(h.finish(), fnv1a(b"foobar"));
    }
}
