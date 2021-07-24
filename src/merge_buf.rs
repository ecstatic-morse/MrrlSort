//! Merge two contiguous sequences using an internal buffer.
//!
//! For now, this is uses a "tape-merge" algorithm.

use crate::Either;
use contracts::*;
use std::cmp::Ordering;
use std::ptr;

/// A contigous chunk of memory consisting of three regions.
///     1. A buffer of elements that can be freely permuted.
///     2. An ordered list of elements (L).
///     3. Another ordered list of elements (R).
///
/// |---buf---|----L----|----R----|
pub struct ContiguousBlockPairAndLeftBuffer<'a, T> {
    pub slice: &'a mut [T],
    pub buf_len: usize,
    pub left_len: usize,
}

impl<'a, T> ContiguousBlockPairAndLeftBuffer<'a, T> {
    #[debug_requires(left_len > 0)]
    #[debug_requires(buf_len > 0)]
    #[debug_requires(slice.len() >= buf_len)]
    #[debug_requires(slice.len() - buf_len >= left_len)]
    #[debug_requires(slice.len() - buf_len - left_len <= buf_len)] // buf_len >= right_len
    pub fn new(slice: &'a mut [T], buf_len: usize, left_len: usize) -> Self {
        ContiguousBlockPairAndLeftBuffer {
            slice,
            buf_len,
            left_len,
        }
    }

    fn ptr_triple(&mut self) -> (*mut T, Pair<*mut [T]>) {
        let (buf, lr) = self.slice.split_at_mut(self.buf_len);
        let (l, r) = lr.split_at_mut(self.left_len);

        (buf.as_mut_ptr(), Pair::new(l as _, r as _))
    }

    #[cfg(test)]
    fn left_and_right(&self) -> crate::Split<&[T]> {
        crate::Split::new(&self.slice[self.buf_len..], self.left_len)
    }

    /// Merges L and R, placing the merged elements at the start.
    ///
    /// ```text
    /// |-------L//R-------|---buf---|
    /// ```
    #[test_requires(self.left_and_right().is_each_side_sorted_by(cmp))]
    pub fn merge_complete(&mut self, cmp: &mut impl FnMut(&T, &T) -> Ordering, prefer: Either) {
        let fixup = self.merge_inner(cmp, prefer);

        let unmerged = fixup.unmerged.into_inner();

        unsafe {
            reposition(unmerged.as_mut_ptr(), fixup.o, unmerged.len());
        }
    }

    /// Merges L and R, placing the merged elements at the start, until one of the
    /// lists is empty. The remaining elements from the non-empty slice are moved to the end of the
    /// list with the buffer in the middle.
    ///
    /// Returns the number of elements remaining and from which buffer they are from.
    ///
    /// ```text
    /// |-----L//R-----|---buf---|--rem--|
    /// ```
    #[test_requires(self.left_and_right().is_each_side_sorted_by(cmp))]
    pub fn merge_partial(
        &mut self,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) -> Either<usize> {
        let fixup = self.merge_inner(cmp, prefer);

        let end = self.slice.as_mut_ptr_range().end;

        // If the remaining elements are from the left sublist, we need to move them to the end of
        // the array. Remaining elements from the right sublist are already at the end.
        if let Either::Left(l) = fixup.unmerged {
            unsafe { reposition(l.as_mut_ptr(), end.sub(l.len()), l.len()) }
        }

        fixup.unmerged.map(|x| x.len())
    }

    #[inline(always)]
    fn merge_inner(
        &mut self,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) -> MergeFixup<T> {
        let (mut o, Pair { mut l, mut r }) = self.ptr_triple();

        loop {
            let ael = l.as_mut_ptr();
            let bel = r.as_mut_ptr();

            let cmp = unsafe { cmp(&*ael, &*bel) };
            let take_left = if prefer.is_left() {
                cmp.is_le()
            } else {
                cmp.is_lt()
            };

            if take_left {
                unsafe {
                    ptr::swap_nonoverlapping(ael, o, 1);
                    o = o.add(1);
                    l = l.get_unchecked_mut(1..);
                }

                if l.len() == 0 {
                    return MergeFixup {
                        unmerged: Either::Right(r),
                        o,
                    };
                }
            } else {
                unsafe {
                    ptr::swap_nonoverlapping(bel, o, 1);
                    o = o.add(1);
                    r = r.get_unchecked_mut(1..);
                }

                if r.len() == 0 {
                    return MergeFixup {
                        unmerged: Either::Left(l),
                        o,
                    };
                }
            }
        }
    }
}

/// A contigous chunk of memory consisting of three regions.
///     1. An ordered list of elements (L).
///     2. Another ordered list of elements (R).
///     3. A buffer of elements that can be freely permuted.
///
/// |----L----|----R----|---buf---|
#[allow(dead_code)]
pub struct ContiguousBlockPairAndRightBuffer<'a, T> {
    slice: &'a mut [T],

    /// The lengths of the left and right lists respectively.
    len: Pair<usize>,
}

#[allow(dead_code)]
impl<'a, T> ContiguousBlockPairAndRightBuffer<'a, T> {
    #[debug_requires(len.l > 0)]
    #[debug_requires(len.r > 0)]
    #[debug_requires(slice.len() >= len.r)]
    #[debug_requires(slice.len() - len.r >= 2 * len.l)]
    fn new(slice: &'a mut [T], len: Pair<usize>) -> Self {
        ContiguousBlockPairAndRightBuffer { slice, len }
    }

    fn ptr_triple(&mut self) -> (*mut T, Pair<*mut [T]>) {
        let buf_len = self.buf_len();

        let (buf, lr) = self.slice.split_at_mut(buf_len);
        let (l, r) = lr.split_at_mut(self.len.l);

        (
            buf.as_mut_ptr(),
            Pair {
                l: l as _,
                r: r as _,
            },
        )
    }

    fn buf_len(&self) -> usize {
        self.slice.len() - self.len.l - self.len.r
    }

    /// Merges L and R, placing the merged elements at the start.
    ///
    /// ```text
    /// |-------L//R-------|---buf---|
    /// ```
    fn merge_complete(&mut self, cmp: &mut impl FnMut(&T, &T) -> Ordering, prefer: Either) {
        let fixup = self.merge_inner(cmp, prefer);

        let unmerged = fixup.unmerged.into_inner();

        unsafe {
            reposition(unmerged.as_mut_ptr(), fixup.o, unmerged.len());
        }
    }

    /// Merges L and R, placing the merged elements at the start, until one of the
    /// lists is empty. The remaining elements from the non-empty slice are moved to the end of the
    /// list with the buffer in the middle.
    ///
    /// Returns the number of elements remaining and from which buffer they are from.
    ///
    /// ```text
    /// |-----L//R-----|---buf---|--rem--|
    /// ```
    fn merge_partial(
        &mut self,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) -> Either<usize> {
        let fixup = self.merge_inner(cmp, prefer);

        let end = self.slice.as_mut_ptr_range().end;

        // If the remaining elements are from the left sublist, we need to move them to the end of
        // the array. Remaining elements from the right sublist are already at the end.
        if let Either::Left(l) = fixup.unmerged {
            unsafe { reposition(l.as_mut_ptr(), end.sub(l.len()), l.len()) }
        }

        fixup.unmerged.map(|x| x.len())
    }

    #[inline(always)]
    fn merge_inner(
        &mut self,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) -> MergeFixup<T> {
        let (mut o, Pair { mut l, mut r }) = self.ptr_triple();

        loop {
            // Take elements from L when possible, since R is further away from the output
            // buffer.
            let ael = l.as_mut_ptr();
            let bel = r.as_mut_ptr();

            let cmp = unsafe { cmp(&*ael, &*bel) };
            let take_left = if prefer.is_left() {
                cmp.is_le()
            } else {
                cmp.is_lt()
            };

            if take_left {
                unsafe {
                    ptr::swap_nonoverlapping(ael, o, 1);
                    o = o.add(1);
                    l = l.get_unchecked_mut(1..);
                }

                if l.len() == 0 {
                    return MergeFixup {
                        unmerged: Either::Right(r),
                        o,
                    };
                }
            } else {
                unsafe {
                    ptr::swap_nonoverlapping(bel, o, 1);
                    o = o.add(1);
                    r = r.get_unchecked_mut(1..);
                }

                if r.len() == 0 {
                    return MergeFixup {
                        unmerged: Either::Left(l),
                        o,
                    };
                }
            }
        }
    }
}

/// Moves a sequence of `count` values of type `T` beginning at `src` to `dst`, preserving their
/// order.
///
/// Values between `dst` and `dst+count` that are not part of the sequence are moved to the newly
/// vacated slots at `src`, but their relative order is not preserved.
///
/// When the address ranges `[src..src+count]` and `[dst..dst+count]` do not intersect, this is
/// equivalent to `swap_nonoverlapping`.
unsafe fn reposition<T>(mut src: *mut T, mut dst: *mut T, mut count: usize) {
    if count == 0 {
        return;
    }

    let offset = src.offset_from(dst);
    if offset == 0 {
        return;
    }

    let chunk_size = offset.unsigned_abs();
    if chunk_size >= count {
        ptr::swap_nonoverlapping(src, dst, count);
        return;
    }

    if offset < 0 {
        // src < dst
        src = src.add(count);
        dst = dst.add(count);

        while count > chunk_size {
            src = src.sub(chunk_size);
            dst = dst.sub(chunk_size);
            ptr::swap_nonoverlapping(src, dst, chunk_size);
            count -= chunk_size;
        }

        src = src.sub(count);
        dst = dst.sub(count);
    } else {
        // src > dst
        while count > chunk_size {
            ptr::swap_nonoverlapping(src, dst, chunk_size);
            src = src.add(chunk_size);
            dst = dst.add(chunk_size);
            count -= chunk_size;
        }
    }

    ptr::swap_nonoverlapping(src, dst, count);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Pair<T> {
    l: T,
    r: T,
}

impl<T> Pair<T> {
    fn new(l: T, r: T) -> Self {
        Pair { l, r }
    }
}

struct MergeFixup<T> {
    unmerged: Either<*mut [T]>,
    o: *mut T,
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        cmp_ignore_idx, for_each_side, merge_by_naive, merge_by_partial_naive,
        split_vec_sorted_nonempty, KeyAndIndex,
    };
    use crate::Split;
    use proptest::prelude::*;

    const BUF_ELEM: i8 = i8::MIN;

    fn test_left_buf_merge_partial(
        input: Split<Vec<i8>>,
        buf_extra: usize,
    ) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input = Split::map_seq(input, KeyAndIndex::map_vec);

        for_each_side(|prefer| {
            let (l, r) = input.split();
            let (expect, expect_rem) = merge_by_partial_naive(l, r, cmp, prefer);

            let buf_len = r.len() + buf_extra;
            let mut buf = vec![KeyAndIndex::without_index(BUF_ELEM); buf_len];
            buf.extend_from_slice(&input.seq);

            let rem = super::ContiguousBlockPairAndLeftBuffer::new(&mut buf[..], buf_len, input.at)
                .merge_partial(&mut cmp_ignore_idx, prefer);

            // Remainder is from the correct side.
            prop_assert_eq!(expect_rem.is_left(), rem.is_left());

            let expect_rem = expect_rem.into_inner();
            let rem = rem.into_inner();

            // Remainder is correct.
            let rem_start = buf.len() - rem;
            prop_assert_eq!(&buf[rem_start..], &expect_rem);

            // Merge is correct.
            let merge_end = input.len() - rem;
            prop_assert_eq!(&buf[..merge_end], &expect);

            // Buffer elements are together.
            prop_assert!(buf[merge_end..rem_start].iter().all(|&x| x.key == BUF_ELEM));
            Ok(())
        })
    }

    fn test_left_buf_merge_complete(
        input: Split<Vec<i8>>,
        buf_extra: usize,
    ) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input = Split::map_seq(input, KeyAndIndex::map_vec);

        for_each_side(|prefer| {
            let (l, r) = input.split();
            let expect = merge_by_naive(l, r, cmp, prefer);

            let buf_len = r.len() + buf_extra;
            let mut buf = vec![KeyAndIndex::without_index(BUF_ELEM); buf_len];
            buf.extend_from_slice(&input.seq);

            super::ContiguousBlockPairAndLeftBuffer::new(&mut buf[..], buf_len, input.at)
                .merge_complete(cmp, prefer);

            // Merged elements are in order.
            let merge_end = input.len();
            prop_assert_eq!(&buf[..merge_end], &expect);

            // Buffer elements are together.
            prop_assert!(buf[merge_end..].iter().all(|&x| x.key == BUF_ELEM));
            Ok(())
        })
    }

    proptest! {
        #[test]
        fn left_buf_merge_complete(input in split_vec_sorted_nonempty(2..100, 0i8..40), buf_extra in 0usize..4) {
            test_left_buf_merge_complete(input, buf_extra)?;
        }

        #[test]
        fn left_buf_merge_partial(input in split_vec_sorted_nonempty(2..100, 0i8..40), buf_extra in 0usize..4) {
            test_left_buf_merge_partial(input, buf_extra)?;
        }
    }

    #[test]
    fn reposition() {
        for len in 0..30 {
            let v: Vec<u8> = (0u8..(len as u8)).collect();

            for count in 0..len {
                for src in 0..len - count {
                    for dst in 0..len - count {
                        let mut tmp = v.clone();

                        unsafe {
                            super::reposition(
                                tmp.as_mut_ptr().add(src),
                                tmp.as_mut_ptr().add(dst),
                                count,
                            )
                        }

                        assert_eq!(tmp[dst..][..count], v[src..][..count]);

                        // `tmp` is a permutation of `v`
                        tmp.sort();
                        assert_eq!(tmp, v);
                    }
                }
            }
        }
    }
}
