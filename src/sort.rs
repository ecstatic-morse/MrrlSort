//! A block merge sort.

use crate::block::merge::{merge_rolled_blocks_tagged, merge_rolled_blocks_untagged};
use crate::block::roll::{roll_blocks_and_tag, roll_blocks_no_tag, MovementImitationBuffer};
use crate::block::Blocks;
use crate::extract_buf::extract_k_distinct;
use crate::merge_buf::ContiguousBlockPairAndLeftBuffer;
use crate::merge_no_buf::merge_in_place_quadratic;
use crate::{Either, Split, quadratic_sort};
use contracts::*;
use static_assertions::const_assert;
use std::cmp::Ordering;

#[cfg(test)]
use crate::test_utils::is_sorted_by;

const INSERTION_SORT_THRESHOLD: usize = if cfg!(test) { 4 } else { 32 };
const FEW_DISTINCT_THRESHOLD: usize = 4;

const_assert!(INSERTION_SORT_THRESHOLD.is_power_of_two());

pub fn sort<T: Ord>(s: &mut [T]) {
    sort_by(s, Ord::cmp)
}

pub fn sort_by<T>(s: &mut [T], mut cmp: impl FnMut(&T, &T) -> Ordering) {
    let s_len = s.len();

    // If the input is very small, just use a quadratic algorithm and return.
    if s_len < INSERTION_SORT_THRESHOLD {
        quadratic_sort::insertion_sort(s, &mut |a, b| cmp(a, b).is_lt());
        return;
    }

    // Phase 0: Extract an auxiliary buffer filled with distinct elements.
    //
    // The desired auxiliary buffer size is `sqrt(s.len()).next_power_of_two()`.

    let desired_aux_buf_size = auxiliary_buffer_size(s_len);
    let num_distinct = extract_k_distinct(s, desired_aux_buf_size, &mut cmp);

    // If we have very few distinct elements, just do a typical merge sort with our quadratic merge
    // subroutine. Don't bother with internal buffering, block tagging, etc.
    if num_distinct < FEW_DISTINCT_THRESHOLD {
        sort_by_few_distinct(s, cmp);
        return;
    }

    // The auxiliary buffer size must be a power of two, so if we failed to find the desired number
    // of distinct elements, we need to round down.
    let aux_buf_size = if num_distinct == desired_aux_buf_size {
        num_distinct
    } else {
        prev_power_of_two(num_distinct)
    };

    // Sort small runs of elements with a low-overhead quadratic sort.
    // This is faster than starting with a run size of 1.
    for run in s[aux_buf_size..].chunks_mut(INSERTION_SORT_THRESHOLD) {
        quadratic_sort::insertion_sort(run, &mut |a, b| cmp(a, b).is_lt());
    }

    // Phase 1: Merge sorted runs while they are smaller than `aux_buf_size`.
    //
    // At the start of the outer loop, our input looks like this, where `aux_buf_front` and
    // `aux_buf_back` are elements from the auxiliary buffer at the front and back of the input
    // slice respectively.
    //
    //     |-aux_buf_front-|--runs--|-aux_buf_back-|
    //
    // At each iteration of the loop, we carve off `run_size` elements from `aux_buf_front`,
    // which we will use as the auxiliary buffer for that iteration. We then merge runs, two at a
    // time, moving the auxiliary buffer to the end of the array. At the end of that iteration, we
    // have moved `run_size` auxiliary buffer elements from the front to the back, and the size
    // of our sorted runs has doubled.

    let mut aux_buf_front = aux_buf_size;
    let mut run_size = INSERTION_SORT_THRESHOLD;

    while run_size < aux_buf_size {
        let aux_buf_back = aux_buf_size - aux_buf_front;

        debug_assert!(aux_buf_front > run_size);
        aux_buf_front -= run_size;

        let buf_and_runs = &mut s[aux_buf_front..s_len - aux_buf_back];
        merge_runs_subblock(buf_and_runs, run_size, &mut cmp);
        run_size *= 2;
    }

    // Phase 1.5: Merge sorted runs of size `aux_buf_size`.
    //
    // At this point, the run size is equal to the size of the auxiliary buffer.
    //
    // FIXME: The paper does a merge pass with the internal buffer on the right, which moves it
    // back to the start without doing any extra rotations. For maximum efficiency, we need to
    // calculate *exactly* how many buffer elements will not be used during phase 2, and tell
    // `choose_k_distinct` to put them at the back of the input instead of the front.

    // NB: This `if` is necessary for small lists where `INSERTION_SORT_THRESHOLD` >
    // `aux_buf_size`.
    if run_size <= aux_buf_size {
        debug_assert_eq!(run_size, aux_buf_size);
        debug_assert_eq!(aux_buf_front, INSERTION_SORT_THRESHOLD);

        let aux_buf_back = aux_buf_size - aux_buf_front;
        s.rotate_right(aux_buf_back);

        merge_runs_subblock(s, run_size, &mut cmp);

        // Move auxiliary buffer back to the start
        s.rotate_right(run_size);
        run_size *= 2;
    }

    // Phase 2: Merge sorted runs larger than `aux_buf_size`, but smaller than `aux_len.pow(2)`.
    //
    // This is the point where we need to roll and tag blocks before doing a superblock merge. We
    // cannot process runs larger than `aux_buf_size.pow(2) / 2`
    // since those have too many blocks to track in the movement imitation buffer.
    //
    // FIXME: At each step, we left-merge runs, moving the buffer to the end of the input, and
    // then rotate it back to the start for the next step. We could eliminate this rotation by
    // alternating between a left-merge and a right-merge at each step, but this would require a
    // whole new superblock merge subroutine.

    let block_size = aux_buf_size;
    let run_size_limit = block_size * block_size; // FIXME: Can this overflow?

    while run_size < run_size_limit {
        merge_runs_superblock(s, block_size, run_size, &mut cmp);
        run_size *= 2;
    }

    // Phase 3: Merge sorted runs larger than `aux_buf_size.pow(2)`.
    //
    // This only happens when we fail to find the desired number of distinct elements in the input.
    // At this point, we can no longer use the superblock merge strategy, since we would have more
    // blocks in each superblock than elements in the buffer. Luckily, our runs are large and
    // contain many duplicates so each merge takes only linear time.

    let (buf, runs) = s.split_at_mut(block_size);
    while run_size < runs.len() {
        merge_runs_quadratic(buf, runs, run_size, &mut cmp);
        run_size *= 2;
    }

    // Phase 5: Fixup
    //
    // Sort the auxiliary buffer and merge it back into the input.

    quadratic_sort::insertion_sort(&mut s[..block_size], &mut |a, b| cmp(a, b).is_lt());
    merge_in_place_quadratic(Split::new(s, block_size), &mut cmp, Either::Left(()));
}

#[test_requires(buf_and_runs[run_size..].chunks(run_size).all(|s| is_sorted_by(s, cmp)))]
pub fn merge_runs_subblock<T>(
    buf_and_runs: &mut [T],
    run_size: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    // Merge runs, two at a time.
    let step_size = 2 * run_size;
    let window_size = 3 * run_size;

    #[rustfmt::skip]
    let rem_buf_and_runs = for_each_window_with_step(
        buf_and_runs,
        window_size,
        step_size,
        |buf_and_runs| {
            ContiguousBlockPairAndLeftBuffer::new(buf_and_runs, run_size, run_size)
                .merge_complete(cmp, Either::Left(()));
        }
    );

    if rem_buf_and_runs.len() > 2 * run_size {
        // If we have less than 2 full runs, but more than 1, merge the full run with the
        // partial one.
        ContiguousBlockPairAndLeftBuffer::new(rem_buf_and_runs, run_size, run_size)
            .merge_complete(cmp, Either::Left(()));
    } else {
        // If we have less than that, there's no merging to be done.
        // Simply rotate the auxiliary buffer to the end.
        rem_buf_and_runs.rotate_left(run_size);
    }
}

#[test_requires(buf_and_runs[block_size..].chunks(run_size).all(|s| is_sorted_by(s, cmp)))]
pub fn merge_runs_superblock<T>(
    buf_and_runs: &mut [T],
    block_size: usize,
    run_size: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    let mid_blk = run_size / block_size;

    let step_size = 2 * run_size;
    let window_size = 2 * run_size + block_size;

    #[rustfmt::skip]
    let rem_buf_and_runs = for_each_window_with_step(
        buf_and_runs,
        window_size,
        step_size,
        |buf_and_runs| roll_tag_and_merge(Blocks::new(buf_and_runs, block_size), mid_blk, cmp)
    );

    // If there is more than one run remaining, merge them (moving the auxiliary buffer to the
    // end).
    let to_rotate = if rem_buf_and_runs.len() > run_size + block_size {
        // If we don't have a full B block, just do a simple merge. No need to roll blocks.
        if rem_buf_and_runs.len() <= run_size + 2 * block_size {
            ContiguousBlockPairAndLeftBuffer::new(rem_buf_and_runs, block_size, run_size)
                .merge_complete(cmp, Either::Left(()));
        } else {
            let rem_buf_and_runs = Blocks::new(rem_buf_and_runs, block_size);
            roll_tag_and_merge(rem_buf_and_runs, mid_blk, cmp);
        }

        buf_and_runs
    } else {
        let rem_len = rem_buf_and_runs.len();
        let len = buf_and_runs.len();
        &mut buf_and_runs[..len - rem_len + block_size]
    };

    // Rotate the auxiliary buffer back to the start of the input.
    to_rotate.rotate_right(block_size);
}

#[test_requires(runs.chunks(run_size).all(|s| is_sorted_by(s, cmp)))]
pub fn merge_runs_quadratic<T>(
    buf: &mut [T],
    runs: &mut [T],
    run_size: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    // The size of our movement buffer imposes a hard limit on the number of blocks we can track
    // while merging a pair of runs. Under normal circumstances, this would compromise our
    // linearithmic runtime guarantee, since these blocks exceed the size of our internal buffer
    // and must be merged with the quadratic algorithm. However, because we only call this routine
    // when we have less than `sqrt(N)` distinct items, the quadratic merge is fast enough and each
    // level takes linear time.

    let num_blocks = buf.len();
    let block_size = (2 * run_size) / num_blocks;

    let mid_blk = run_size / block_size;

    debug_assert_eq!(run_size % num_blocks, 0);
    debug_assert!(
        block_size > buf.len(),
        "Otherwise no reason to do the quadratic version"
    );

    let mut pairs = runs.chunks_exact_mut(2 * run_size);
    while let Some(pair) = pairs.next() {
        let pair_blocks = Blocks::new(pair, block_size);

        let mvmt = MovementImitationBuffer::new(buf, cmp);
        let mvmt = roll_blocks_no_tag(rbrw!(pair_blocks), mid_blk, mvmt, cmp);

        merge_rolled_blocks_untagged(pair_blocks, mvmt, cmp);
    }

    let rem = pairs.into_remainder();
    if rem.len() > run_size {
        // If we don't have a full B block, just do a quadratic merge. No need to roll blocks.
        if rem.len() <= run_size + block_size {
            merge_in_place_quadratic(Split::new(rem, run_size), cmp, Either::Left(()));
            return;
        }

        let rem_end = rem.len() - (rem.len() % block_size);

        let mvmt = MovementImitationBuffer::new(buf, cmp);
        let pair_blocks_exact = Blocks::new(&mut rem[..rem_end], block_size);
        let mvmt = roll_blocks_no_tag(pair_blocks_exact, mid_blk, mvmt, cmp);

        let pair_blocks = Blocks::new(rem, block_size);
        merge_rolled_blocks_untagged(pair_blocks, mvmt, cmp);
    }
}

/// A typical bottom-up merge sort using an in-place, quadratic merge.
///
/// This is only used when we have very few distinct elements (less than `INSERTION_SORT_THRESHOLD`).
/// At that point, it's a waste of time to do superblock merges with an internal buffer.
fn sort_by_few_distinct<T>(s: &mut [T], mut cmp: impl FnMut(&T, &T) -> Ordering) {
    for run in s.chunks_mut(INSERTION_SORT_THRESHOLD) {
        quadratic_sort::insertion_sort(run, &mut |a, b| cmp(a, b).is_lt());
    }

    let mut run_size = INSERTION_SORT_THRESHOLD;
    while run_size < s.len() {
        let mut pairs = s.chunks_exact_mut(2 * run_size);
        while let Some(pair) = pairs.next() {
            merge_in_place_quadratic(Split::new(pair, run_size), &mut cmp, Either::Left(()))
        }

        let rem = pairs.into_remainder();
        if rem.len() > run_size {
            merge_in_place_quadratic(Split::new(rem, run_size), &mut cmp, Either::Left(()))
        }

        run_size *= 2;
    }
}

/// Calls `f` for successive subslices of `s` (with size `window_size`), advancing by `step_size`
/// each time. Returns a subslice containing any elements at the end of the slice that were not
/// part of a window.
///
/// We would prefer to use a `WindowsMut` iterator, but one cannot be implemented without streaming
/// iterators.
#[debug_requires(step_size <= window_size)]
fn for_each_window_with_step<T>(
    s: &mut [T],
    window_size: usize,
    step_size: usize,
    mut f: impl FnMut(&mut [T]),
) -> &mut [T] {
    let mut start = 0;

    while start + window_size < s.len() {
        let window = &mut s[start..][..window_size];
        f(window);
        start += step_size;
    }

    if start < s.len() {
        &mut s[start..]
    } else {
        &mut []
    }
}

fn roll_tag_and_merge<T>(
    buf_and_runs: Blocks<&mut [T]>,
    mid_blk: usize,
    mut cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    let block_size = buf_and_runs.blk_len;
    let (mvmt, runs) = buf_and_runs.seq.split_at_mut(block_size);

    let roll_input_end = runs.len() - (runs.len() % block_size);
    let rslt = roll_blocks_and_tag(
        Blocks::new(&mut runs[..roll_input_end], buf_and_runs.blk_len),
        mid_blk,
        MovementImitationBuffer::new(mvmt, &mut cmp),
        &mut cmp,
    );

    merge_rolled_blocks_tagged(buf_and_runs, rslt, &mut cmp);
}

#[ensures(ret.is_power_of_two())]
fn auxiliary_buffer_size(len: usize) -> usize {
    fast_approx_isqrt(len).next_power_of_two()
}

fn fast_approx_isqrt(x: usize) -> usize {
    if x == 0 {
        return x;
    }

    let scale = usize::BITS - x.leading_zeros();
    let half = scale / 2;
    let guess = ((x >> half) + (1 << half)) / 2;
    guess
}

fn prev_power_of_two(x: usize) -> usize {
    let n = usize::BITS - x.leading_zeros();
    if n == 0 {
        0
    } else {
        1 << (n - 1)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{cmp_ignore_idx, is_sorted, KeyAndIndex};
    use proptest::prelude::*;

    #[test]
    fn grailsort_oneshot() {
        let input = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 3,
            7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 13, 0, 14, 16, 15, 9, 5, 10, 11, 6, 17,
        ];

        let mut input = KeyAndIndex::map_vec(input);
        let mut expected = input.clone();

        // We can use a faster, unstable sort because the index is included in the comparator.
        expected.sort_unstable();
        super::sort_by(&mut input, cmp_ignore_idx);

        assert_eq!(input, expected);
    }

    fn test_grailsort<T: std::fmt::Debug + Ord + Clone>(
        input: Vec<T>,
    ) -> Result<(), TestCaseError> {
        let mut input = KeyAndIndex::map_vec(input);
        let mut expected = input.clone();

        super::sort_by(&mut input, cmp_ignore_idx);

        if !is_sorted(&input) {
            // Only sort `expected` if `input` is in the wrong order. This runs faster while
            // still giving a nice assertion message.
            expected.sort_unstable(); // See above.
            prop_assert_eq!(input, expected);
            unreachable!("`input` not sorted but `sort_unstable` returned the same result???");
        }

        Ok(())
    }

    const THRESHOLD: u8 = super::FEW_DISTINCT_THRESHOLD as u8;
    const THRESHOLD_PLUS_EPSILON: u8 = THRESHOLD + THRESHOLD / 2;

    proptest! {
        #[test]
        fn prev_power_of_two(x in 0usize..usize::MAX) {
            let expect = if x.is_power_of_two() {
                x
            } else {
                x.checked_next_power_of_two().map_or(1 << usize::BITS - 1, |x| x >> 1)
            };

            prop_assert_eq!(super::prev_power_of_two(x), expect);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000_000))]

        #[test]
        fn grailsort_few_distinct(input in proptest::collection::vec(0u8..THRESHOLD, 0..2048)) {
            test_grailsort(input)?;
        }

        #[test]
        fn grailsort_some_distinct(input in proptest::collection::vec(0u8..THRESHOLD_PLUS_EPSILON, 0..2048)) {
            test_grailsort(input)?;
        }

        #[test]
        fn grailsort_many_distinct(input in proptest::collection::vec(0u8..(THRESHOLD * 6), 0..2048)) {
            test_grailsort(input)?;
        }
    }
}
