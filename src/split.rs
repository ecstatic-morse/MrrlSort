//! Wrappers around a slice to allow alternate indexing, midpoints, etc.

#[allow(unused)]
use std::cmp::Ordering;

/// A sequence (`S`) divided in two at index `I`.
#[derive(Clone, Copy, Debug)]
pub struct Split<S, I = usize> {
    pub seq: S,
    pub at: I,
}

impl<S> Split<S> {
    pub fn split<T>(&self) -> (&[T], &[T])
    where
        S: AsRef<[T]>,
    {
        self.seq.as_ref().split_at(self.at)
    }

    #[cfg(test)]
    pub fn is_each_side_sorted_by<T>(&self, cmp: &mut impl FnMut(&T, &T) -> Ordering) -> bool
    where
        S: AsRef<[T]>,
    {
        use crate::test_utils::is_sorted_by;

        let (l, r) = self.split();
        is_sorted_by(l, cmp) && is_sorted_by(r, cmp)
    }

    #[cfg(test)]
    pub fn split_mut<T>(&mut self) -> (&mut [T], &mut [T])
    where
        S: AsMut<[T]>,
    {
        self.seq.as_mut().split_at_mut(self.at)
    }
}

impl<S, I> Split<S, I> {
    pub fn new(seq: S, at: I) -> Self {
        Split { seq, at }
    }

    #[cfg(test)]
    pub fn len<T>(&self) -> usize
    where
        S: AsRef<[T]>,
    {
        self.seq.as_ref().len()
    }

    #[cfg(test)]
    #[allow(unused)]
    pub fn as_ref<T>(&self) -> Split<&T, I>
    where
        T: ?Sized,
        S: AsRef<T>,
        I: Copy,
    {
        Split {
            seq: self.seq.as_ref(),
            at: self.at,
        }
    }

    #[cfg(test)]
    pub fn as_mut<T>(&mut self) -> Split<&mut T, I>
    where
        T: ?Sized,
        S: AsMut<T>,
        I: Copy,
    {
        Split {
            seq: self.seq.as_mut(),
            at: self.at,
        }
    }

    #[cfg(test)]
    pub fn map_seq<R>(self, f: impl FnOnce(S) -> R) -> Split<R, I> {
        Split {
            seq: f(self.seq),
            at: self.at,
        }
    }
}
