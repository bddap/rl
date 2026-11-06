//! The one byte cursor every wire decoder reads through.

/// A read ran past the end of the buffer. Each decoder maps it into its own error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Truncated;

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    pub(crate) fn take<const N: usize>(&mut self) -> Result<[u8; N], Truncated> {
        let end = self.at.checked_add(N).ok_or(Truncated)?;
        let slice = self.buf.get(self.at..end).ok_or(Truncated)?;
        self.at = end;
        Ok(slice.try_into().expect("slice length checked above"))
    }

    pub(crate) fn byte(&mut self) -> Result<u8, Truncated> {
        Ok(self.take::<1>()?[0])
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.at
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.remaining() == 0
    }
}
