//! Merge two contiguous sequences **without** using an internal buffer.

use crate::search::{binary_search, exp_search, rexp_search};
use crate::{Either, Split};
use contracts::*;
use std::cmp::Ordering;

/// Merges two sorted slices that are contiguous in memory into a single, sorted slice without
/// using additional storage.
///
/// # Runtime
///
/// If `M` is the size of the larger sequence and `N` is the size of the smaller one,
/// this operation requires `O(N^2 + M)` writes and `O(N log M)` comparisons.
///
/// # Algorithm
///
/// For now, this uses a "simple binary" algorithm, described below.
///
/// Let's examine the case where the left array (`L`) is smaller than the right one (`R`).
/// Start with the leftmost element of `L`. First, find the number of
/// elements in `L` equal to the leftmost one. Call this `ndupes`. Next, binary search for the
/// index the aforementioned element would occupy in `R`. Call this index `p`. All elements in
/// `R[..p]` belong before `A` in the final, merged array. Shift them into place by right-rotating
/// the slice formed by concatenating of `L` with the first `p` elements of `R` by `p` positions.
///
/// `p+ndupes` elements are now in their final position (`p` elements from `R` and `ndupes` from `L`).
/// Recurse on the remaining elements of `L` and `R`, which are still contigous in memory, until
/// one is empty. The arrays are now merged.
///
/// This mirror image of this procedure is used to handle the case where the smaller array is on
/// the right. That is, we start with the *rightmost* element of the smaller array and
/// *left*-rotate the concatenated slice.  I've illustrated both versions of the algorithm below.
///
/// ```text
/// Small array on left:         Small array on right:
///                     ndupes=2
///  L     R                      L          R
/// [99..][678, ,,,,]            [,,,, ,789][..66]
///
///        p=3                          p=3
///  v    |---v                        v---|    v
/// [99..][678, ,,,,]            [,,,, ,789][..66]
///
/// rotate_right(3)                     rotate_left(3)
/// |--------|                          |--------o
/// [99..][678, ,,,,]            [,,,, ,789][..66]
///
///       L   R                   L       R
/// 67899[..][, ,,,,]            [,,,, ,][..]66789
/// ```
#[test_requires(pair.is_each_side_sorted_by(cmp))]
pub(crate) fn merge_in_place_quadratic<T>(
    pair: Split<&mut [T]>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) {
    fn small_on_left<T>(
        pair: Split<&mut [T]>,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) {
        let Split {
            seq: mut slice,
            at: mut mid,
        } = pair;

        while mid != 0 && mid < slice.len() {
            // SAFETY: See comments below.
            unsafe {
                let lil = slice.get_unchecked(..mid);
                let big = slice.get_unchecked(mid..);

                // From the definitions of `lil` and `big`, we have:
                //      lil.len() + big.len() == slice.len()
                //      lil.len() == mid
                //      big.len() == slice.len() - mid
                // And from the loop condition:
                //      lil.len() > 0
                //      big.len() > 0

                let (lil_el, lil_tail) = lil.split_first().unwrap_unchecked();
                let ndupes = 1 + exp_search(lil_tail, |x| cmp(x, lil_el).is_eq()); // ndupes <= lil.len()
                let p = binary_search(big, |x| is_lt(cmp(x, lil_el), prefer.flip())); // p <= big.len()

                // From above:
                //      p <= big.len()
                //      mid == lil.len()
                //      big.len() + lil.len() == slice.len()
                // It follows that:
                //      p + mid <= slice.len()
                let to_rotate = slice.get_unchecked_mut(..mid + p); // to_rotate.len() == mid + p

                // The following is trivially true since `mid+p` is the length of `to_rotate` and
                // `mid > 0`. However, LLVM 11 still cannot optimize away the bounds check in
                // `rotate` without the explicit `assume`.
                std::intrinsics::assume(p < to_rotate.len());
                to_rotate.rotate_right(p);

                // From above:
                //      p <= big.len()
                //      ndupes <= lil.len()
                //      slice.len() == big.len() + lil.len()
                // It follows that:
                //      p + ndupes <= slice.len()
                slice = slice.get_unchecked_mut(p + ndupes..);

                // This cannot overflow because:
                //      lil.len() == mid
                //      ndupes <= lil.len()
                mid -= ndupes;
            }
        }
    }

    fn small_on_right<T>(
        pair: Split<&mut [T]>,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
        prefer: Either,
    ) {
        let Split {
            seq: mut slice,
            at: mut mid,
        } = pair;

        // A = slice[mid..], B = slice[..mid]
        // INVARIANT: A.len() > 0, B.len() > 0
        while mid != 0 && mid < slice.len() {
            // SAFETY: See comments below
            unsafe {
                let big = slice.get_unchecked(..mid);
                let lil = slice.get_unchecked(mid..);

                // From the definitions of `big` and `lil`, we have:
                //      lil.len() + big.len() == slice.len()
                //      lil.len() == slice.len() - mid
                //      big.len() == mid
                // And from the loop condition:
                //      lil.len() > 0
                //      big.len() > 0

                let (lil_el, lil_tail) = lil.split_last().unwrap_unchecked();

                // ndupes <= lil.len()    (lil_tail has one less element than lil)
                let ndupes = 1 + rexp_search(lil_tail, |x| cmp(x, lil_el).is_eq());

                // mid - p <= big.len()
                let mid_minus_p = binary_search(big, |x| is_lt(cmp(x, lil_el), prefer));

                // mid - (mid - p) = mid - mid + p = p
                let p = mid - mid_minus_p;

                // From above:
                //      mid - p <= big.len()
                //      big.len() < slice.len()
                // Therefore:
                //      mid - p < slice.len()
                let to_rotate = slice.get_unchecked_mut(mid_minus_p..);

                // From the definition of `to_rotate`, we have:
                //     to_rotate.len() = slice.len() - (mid - p)
                //                     = slice.len() - mid + p
                //                     = lil.len() + p
                //
                // And, since `lil.len() > 0`:
                //      to_rotate.len() > p
                std::intrinsics::assume(p < to_rotate.len());
                to_rotate.rotate_left(p);

                // From above:
                //      p <= big.len()
                //      ndupes <= lil.len()
                //      slice.len() == big.len() + lil.len()
                // It follows that:
                //      p + ndupes <= slice.len()
                // Which means the this subtraction cannot overflow.
                let len = slice.len();
                slice = slice.get_unchecked_mut(..len - p - ndupes);

                mid = mid_minus_p;
            }
        }
    }

    if pair.at / 2 > pair.seq.len() {
        small_on_right(pair, cmp, prefer);
    } else {
        small_on_left(pair, cmp, prefer);
    }
}

pub(crate) fn merge_in_place_quadratic_partial<T>(
    pair: Split<&mut [T]>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) -> Either<usize> {
    let ret;

    // FIXME: Calculate this as part of the actual merge instead of seperately
    let (l, r) = pair.split();
    match (l.last(), r.last()) {
        (None, _) => return Either::Right(r.len()),
        (_, None) => return Either::Left(l.len()),

        (Some(tl), Some(tr)) => {
            if is_lt(cmp(tl, tr), prefer) {
                let n = binary_search(r, |x| !is_lt(cmp(tl, x), prefer));
                ret = Either::Right(r.len() - n);
            } else {
                let n = binary_search(l, |x| is_lt(cmp(x, tr), prefer));
                ret = Either::Left(l.len() - n);
            }
        }
    }

    merge_in_place_quadratic(pair, cmp, prefer);
    ret
}

fn is_lt(ord: Ordering, prefer: Either) -> bool {
    if prefer.is_left() {
        ord.is_le()
    } else {
        ord.is_lt()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{self, cmp_ignore_idx, for_each_side, split_vec_sorted, KeyAndIndex};
    use crate::Split;
    use proptest::prelude::*;
    use std::fmt::Debug;

    fn test_merge<T: Clone + Debug + Ord>(input: Split<Vec<T>>) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input = Split::map_seq(input, KeyAndIndex::map_vec);

        for_each_side(|prefer| {
            let (l, r) = input.split();
            let expected = test_utils::merge_by_naive(l, r, cmp, prefer);

            let mut actual = input.clone();
            let actual_mut: Split<&mut [_]> = actual.as_mut();
            super::merge_in_place_quadratic(actual_mut, cmp, prefer);

            prop_assert_eq!(actual.seq, expected);
            Ok(())
        })
    }

    fn test_merge_partial<T: Clone + Debug + Ord>(
        input: Split<Vec<T>>,
    ) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input = Split::map_seq(input, KeyAndIndex::map_vec);

        for_each_side(|prefer| {
            let (l, r) = input.split();
            let (mut expected, expected_rem) =
                test_utils::merge_by_partial_naive(l, r, cmp, prefer);

            let mut actual = input.clone();
            let actual_mut: Split<&mut [_]> = actual.as_mut();
            let actual_rem = super::merge_in_place_quadratic_partial(actual_mut, cmp, prefer);

            prop_assert_eq!(actual_rem, expected_rem.as_ref().map(|v| v.len()));

            expected.extend(expected_rem.into_inner());
            prop_assert_eq!(actual.seq, expected);
            Ok(())
        })
    }

    #[test]
    fn merge_oneshot() {
        prop_unwrap!(test_merge(Split::new(
            vec![1, 4, 4, 4, 9, 1, 3, 4, 7, 7],
            5
        )));
    }

    #[test]
    fn merge_partial_oneshot() {
        prop_unwrap!(test_merge_partial(Split::new(
            vec![1, 4, 4, 4, 9, 1, 3, 4, 7, 7],
            5
        )));
    }

    proptest! {
        #[test]
        fn merge(input in split_vec_sorted(0..100, 0u8..10)) {
            test_merge(input)?;
        }

        #[test]
        fn merge_partial(input in split_vec_sorted(0..100, 0u8..10)) {
            test_merge_partial(input)?;
        }
    }
}
