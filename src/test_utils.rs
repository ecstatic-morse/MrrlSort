use crate::block::{Blk, Blocks, SplitBlocks};
use crate::{Either, Split};
use proptest::prelude::*;
use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt;
use std::ops::Range;

/// Calls `f` with both variants of `Either`.
pub fn for_each_side<T, E>(mut f: impl FnMut(Either) -> Result<T, E>) -> Result<T, E> {
    f(Either::Left(()))?;
    f(Either::Right(()))
}

#[cfg(test)]
macro_rules! prop_unwrap {
    ($e:expr) => {
        match $e {
            Err(::proptest::prelude::TestCaseError::Reject(s)) => {
                panic!("Helper function rejected input: {}", s)
            }
            Err(::proptest::prelude::TestCaseError::Fail(s)) => panic!("{}", s),
            Ok(x) => x,
        }
    };
}

pub fn is_sorted_by<T>(s: &[T], cmp: &mut impl FnMut(&T, &T) -> Ordering) -> bool {
    s.windows(2).all(|pair| cmp(&pair[0], &pair[1]).is_le())
}

pub fn is_sorted<T: Ord>(s: &[T]) -> bool {
    is_sorted_by(s, &mut Ord::cmp)
}

/// Creates a `Blocks<Vec<T>>` whose length may or may not be a multiple of the block size.
pub fn vec_blocks<S>(
    num_blocks: Range<usize>,
    blk_len: Range<usize>,
    values: S,
) -> impl Strategy<Value = Blocks<Vec<S::Value>>>
where
    S: Clone + Strategy,
    S::Value: Clone,
{
    (num_blocks, blk_len)
        .prop_flat_map(move |(num_blocks, blk_len)| {
            let len = blk_len * num_blocks;
            let range = len..len + blk_len;
            (
                proptest::collection::vec(values.clone(), range),
                Just(blk_len),
            )
        })
        .prop_map(|(seq, blk_len)| Blocks::new(seq, blk_len))
}

/// Creates a `Blocks<Vec<T>>` whose length is a multiple of the block size.
pub fn vec_blocks_exact<S>(
    num_blocks: Range<usize>,
    blk_len: Range<usize>,
    values: S,
) -> impl Strategy<Value = Blocks<Vec<S::Value>>>
where
    S: Clone + Strategy,
    S::Value: Clone,
{
    (num_blocks, blk_len)
        .prop_flat_map(move |(num_blocks, blk_len)| {
            let len = blk_len * num_blocks;
            (
                proptest::collection::vec(values.clone(), len..len + 1),
                Just(blk_len),
            )
        })
        .prop_map(|(seq, blk_len)| Blocks::new(seq, blk_len))
}

pub fn vec_blocks_split<S>(
    num_blocks: Range<usize>,
    blk_len: Range<usize>,
    values: S,
) -> impl Strategy<Value = SplitBlocks<Vec<S::Value>>>
where
    S: Clone + Strategy,
    S::Value: Clone,
{
    vec_blocks(num_blocks, blk_len, values)
        .prop_flat_map(|v| {
            let n = v.num_blocks();
            (Just(v), 0..=n)
        })
        .prop_map(|(v, mid_blk)| Split::new(v, Blk(mid_blk)))
}

pub fn vec_blocks_exact_split<S>(
    num_blocks: Range<usize>,
    blk_len: Range<usize>,
    values: S,
) -> impl Strategy<Value = SplitBlocks<Vec<S::Value>>>
where
    S: Clone + Strategy,
    S::Value: Clone,
{
    vec_blocks_exact(num_blocks, blk_len, values)
        .prop_flat_map(|v| {
            let n = v.num_blocks();
            (Just(v), 0..=n)
        })
        .prop_map(|(v, mid_blk)| Split::new(v, Blk(mid_blk)))
}

pub fn nonempty<T: std::fmt::Debug>(
    strat: impl Strategy<Value = SplitBlocks<Vec<T>>>,
) -> impl Strategy<Value = SplitBlocks<Vec<T>>> {
    strat.prop_map(|mut v| {
        if v.seq.num_blocks() < 2 {
            panic!("Input too small to be nonempty");
        }

        if v.at == Blk(0) {
            v.at = Blk(1);
        } else if v.at == Blk(v.seq.num_blocks()) {
            v.at = Blk(v.at.0 - 1);
        }

        v
    })
}

pub fn sorted<T: Ord + std::fmt::Debug>(
    strat: impl Strategy<Value = SplitBlocks<Vec<T>>>,
) -> impl Strategy<Value = SplitBlocks<Vec<T>>> {
    strat.prop_map(|mut v| {
        let (l, r) = v.seq.seq.split_at_mut(v.at.0 * v.seq.blk_len);
        l.sort_unstable();
        r.sort_unstable();
        v
    })
}

pub fn split_vec<S>(
    len: impl Into<proptest::collection::SizeRange>,
    values: S,
) -> impl Strategy<Value = Split<Vec<S::Value>>>
where
    S: Strategy,
    S::Value: Clone,
{
    proptest::collection::vec(values, len)
        .prop_flat_map(|v| {
            let len = v.len();
            (Just(v), 0..=len)
        })
        .prop_map(|(seq, at)| Split { seq, at })
}

pub fn split_vec_sorted<S>(
    len: impl Into<proptest::collection::SizeRange>,
    values: S,
) -> impl Strategy<Value = Split<Vec<S::Value>>>
where
    S: Strategy,
    S::Value: Clone + Ord,
{
    split_vec(len, values).prop_map(|mut out| {
        let (a, b) = out.split_mut();
        a.sort_unstable();
        b.sort_unstable();
        out
    })
}

pub fn split_vec_sorted_nonempty<S>(
    len: std::ops::Range<usize>,
    values: S,
) -> impl Strategy<Value = Split<Vec<S::Value>>>
where
    S: Strategy,
    S::Value: Clone + Ord,
{
    assert!(len.end - len.start >= 2);

    split_vec_sorted(len, values).prop_map(|mut buf| {
        if buf.at == 0 {
            buf.at += 1;
        } else if buf.at == buf.seq.len() {
            buf.at -= 1;
        }

        buf
    })
}

/// An element coupled with its index in some container.
///
/// Used to test the stability of sorting algorithms.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyAndIndex<T> {
    pub key: T,
    pub idx: u32, // `u32` cuts the amount of memory required in half on 64-bit systems.
}

impl<T: fmt::Debug> fmt::Debug for KeyAndIndex<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn to_string_subscript(mut i: usize) -> String {
            if i == 0 {
                return "₀".to_owned();
            }

            let mut ret = Vec::new();
            while i != 0 {
                let digit = i % 10;
                ret.push(char::from_u32('₀' as u32 + digit as u32).unwrap());
                i /= 10;
            }

            ret.into_iter().rev().collect()
        }

        let sub = to_string_subscript(self.idx as usize);
        write!(f, "{:?}{}", self.key, sub)
    }
}

impl<T> KeyAndIndex<T> {
    pub fn new(key: T, idx: usize) -> Self {
        let idx = idx.try_into().expect("Index overflowed a u32");
        KeyAndIndex { key, idx }
    }

    pub fn without_index(key: T) -> Self {
        KeyAndIndex { key, idx: u32::MAX }
    }

    pub fn map_vec(v: Vec<T>) -> Vec<KeyAndIndex<T>> {
        v.into_iter()
            .enumerate()
            .map(|(idx, key)| KeyAndIndex::new(key, idx))
            .collect()
    }
}

pub fn cmp_ignore_idx<T: Ord>(a: &KeyAndIndex<T>, b: &KeyAndIndex<T>) -> Ordering {
    cmp_by_ignore_idx(a, b, &mut Ord::cmp)
}

pub fn cmp_by_ignore_idx<T>(
    a: &KeyAndIndex<T>,
    b: &KeyAndIndex<T>,
    cmp: impl FnOnce(&T, &T) -> Ordering,
) -> Ordering {
    cmp(&a.key, &b.key)
}

pub fn merge_by_naive<T: Clone>(
    l: &[T],
    r: &[T],
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) -> Vec<T> {
    let mut ret = Vec::with_capacity(l.len() + r.len());

    let mut i = 0;
    let mut j = 0;

    while i < l.len() && j < r.len() {
        let cmp = cmp(&l[i], &r[j]);
        if cmp.is_lt() || cmp.is_eq() && prefer.is_left() {
            ret.push(l[i].clone());
            i += 1;
        } else {
            ret.push(r[j].clone());
            j += 1;
        }
    }

    if i == l.len() {
        ret.extend_from_slice(&r[j..]);
    } else {
        assert_eq!(j, r.len());
        ret.extend_from_slice(&l[i..]);
    }

    ret
}

pub fn merge_by_partial_naive<T: Clone>(
    l: &[T],
    r: &[T],
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) -> (Vec<T>, Either<Vec<T>>) {
    let mut ret = Vec::with_capacity(l.len() + r.len());

    let mut i = 0;
    let mut j = 0;

    while i < l.len() && j < r.len() {
        let cmp = cmp(&l[i], &r[j]);
        if cmp.is_lt() || cmp.is_eq() && prefer.is_left() {
            ret.push(l[i].clone());
            i += 1;
        } else {
            ret.push(r[j].clone());
            j += 1;
        }
    }

    if i == l.len() {
        (ret, Either::Right(r[j..].to_vec()))
    } else {
        assert_eq!(j, r.len());
        (ret, Either::Left(l[i..].to_vec()))
    }
}
