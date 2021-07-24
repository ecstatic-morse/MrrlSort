//! Late-phase routines for blocks of elements.

use contracts::*;
use std::ptr;

pub mod merge;
pub mod roll;

/// Index a `Blocks` by blocks (instead of elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Blk(pub usize);

#[allow(unused)]
pub type SplitBlocks<T> = crate::Split<Blocks<T>, Blk>;

/// A contiguous sequence of elements divided into chunks of a certain size.
///
/// The number of elements is not necessarily a multiple of the `blk_len`.
#[derive(Clone, Copy, Debug)]
pub struct Blocks<S> {
    pub seq: S,
    pub blk_len: usize,
}

fn tail_idx(i: usize, blk_len: usize) -> usize {
    (i * blk_len) + blk_len - 1
}

impl<S> Blocks<S> {
    #[requires(blk_len > 0)]
    pub fn new(seq: S, blk_len: usize) -> Blocks<S> {
        Blocks { seq, blk_len }
    }
}

impl<S> Blocks<S> {
    /// Returns the number of **full** blocks.
    pub fn num_blocks<T>(&self) -> usize
    where
        S: AsRef<[T]>,
    {
        self.seq.as_ref().len() / self.blk_len
    }

    /// Returns `true` if the size of the underlying sequence is a multiple of the block size.
    pub fn is_exact<T>(&self) -> bool
    where
        S: AsRef<[T]>,
    {
        self.seq.as_ref().len() % self.blk_len == 0
    }

    #[requires(i < self.num_blocks())]
    pub fn head<T>(&self, i: usize) -> &T
    where
        S: AsRef<[T]>,
    {
        &self.seq.as_ref()[i * self.blk_len]
    }

    #[requires(i < self.num_blocks())]
    pub fn tail<T>(&self, i: usize) -> &T
    where
        S: AsRef<[T]>,
    {
        &self.seq.as_ref()[tail_idx(i, self.blk_len)]
    }

    #[requires(i < self.num_blocks())]
    pub fn block<T>(&self, i: usize) -> &[T]
    where
        S: AsRef<[T]>,
    {
        let blk_len = self.blk_len;
        &self.seq.as_ref()[i * blk_len..][..blk_len]
    }

    #[allow(unused)]
    pub fn iter<'a, T: 'a>(&'a self) -> impl 'a + Iterator<Item = &[T]>
    where
        S: AsRef<[T]>,
    {
        self.seq.as_ref().chunks_exact(self.blk_len)
    }

    pub fn tails_enumerated<'a, T: 'a>(&'a self) -> impl 'a + Iterator<Item = (usize, &T)>
    where
        S: AsRef<[T]>,
    {
        self.seq
            .as_ref()
            .iter()
            .skip(self.blk_len - 1)
            .step_by(self.blk_len)
            .enumerate()
    }

    #[requires(i < self.num_blocks())]
    pub fn block_mut<'a, T>(&'a mut self, i: usize) -> &'a mut [T]
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let blk_len = self.blk_len;
        &mut self.seq.as_mut()[i * blk_len..][..blk_len]
    }

    #[allow(unused)]
    pub fn iter_mut<'a, T: 'a>(&'a mut self) -> impl 'a + Iterator<Item = &mut [T]>
    where
        S: AsMut<[T]>,
    {
        self.seq.as_mut().chunks_exact_mut(self.blk_len)
    }

    /// Swaps the head of the block with the first index with the tail of the block with the second.
    #[requires(head_blk < self.num_blocks())]
    #[requires(tail_blk < self.num_blocks())]
    #[requires(self.blk_len >= 2, "Block head and block tail should be disjoint")]
    pub fn swap_head_tail<T>(&mut self, head_blk: usize, tail_blk: usize)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let tail = tail_idx(tail_blk, self.blk_len);
        let head = head_blk * self.blk_len;

        self.seq.as_mut().swap(head, tail);
    }

    #[requires(i < self.num_blocks())]
    #[requires(j < self.num_blocks())]
    pub fn swap_blocks<T>(&mut self, i: usize, j: usize)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        if i == j {
            return;
        }

        let ptr = self.seq.as_mut().as_mut_ptr();
        unsafe {
            let a = ptr.add(i * self.blk_len);
            let b = ptr.add(j * self.blk_len);

            ptr::swap_nonoverlapping(a, b, self.blk_len)
        }
    }

    #[cfg(test)]
    pub fn map_seq<R>(self, f: impl FnOnce(S) -> R) -> Blocks<R> {
        Blocks {
            seq: f(self.seq),
            blk_len: self.blk_len,
        }
    }

    #[cfg(test)]
    pub fn as_mut<R: ?Sized>(&mut self) -> Blocks<&mut R>
    where
        S: AsMut<R>,
    {
        Blocks {
            seq: self.seq.as_mut(),
            blk_len: self.blk_len,
        }
    }
}

impl<'a, S: ?Sized> Blocks<&'a mut S> {
    pub fn as_shared(self) -> Blocks<&'a S> {
        Blocks {
            seq: &*self.seq,
            blk_len: self.blk_len,
        }
    }
}

/*
impl<T> std::ops::Index<Blk> for Blocks<T>
where
    T: AsRef<[T]>,
{
    type Output = [T];

    fn index(&self, index: Blk) -> &Self::Output {
        let start = self.blk_len * index.0;
        let end = start + self.blk_len;

        &self.seq.as_ref()[start..end]
    }
}

impl<S, T> std::ops::IndexMut<Blk> for Blocks<S> where S: AsMut<[T]> {
    fn index_mut(&mut self, index: Blk) -> &mut Self::Output {
        let start = self.blk_len * index.0;
        let end = start + self.blk_len;

        &mut self.seq.as_mut()[start..end]
    }
}
*/
