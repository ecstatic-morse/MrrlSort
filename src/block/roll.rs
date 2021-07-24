//! Subroutines for rolling and tagging blocks prior to a block merge.

use std::cmp::Ordering;
use std::ops::{self, Range};

use contracts::*;

use super::Blocks;

#[cfg(test)]
use crate::test_utils::is_sorted_by;

/// A newtype that indicates this slice will be used as a movement imitation buffer.
///
/// For now, this is just a marker to ensure that arguments are passed in the correct order.
/// However, it might be better to encode some invariants (e.g., all elements are distinct, the slice
/// is sorted on creation, etc.) as well.
pub struct MovementImitationBuffer<'a, T>(&'a mut [T]);

impl<'a, T> MovementImitationBuffer<'a, T> {
    pub fn new(slice: &'a mut [T], cmp: &mut impl FnMut(&T, &T) -> Ordering) -> Self {
        slice.sort_unstable_by(&mut *cmp);

        #[cfg(test)]
        assert!(slice.windows(2).all(|w| cmp(&w[0], &w[1]).is_ne()));

        MovementImitationBuffer(slice)
    }

    pub fn freeze(self) -> MovementImitationResult<'a, T> {
        MovementImitationResult(self.0)
    }
}

impl<'a, T> ops::Deref for MovementImitationBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

pub struct MovementImitationResult<'a, T>(&'a [T]);

impl<'a, T> MovementImitationResult<'a, T> {
    pub fn subslice(self, range: Range<usize>) -> Self {
        MovementImitationResult(&self.0[range])
    }
}

impl<'a, T> ops::Deref for MovementImitationResult<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// The number of A blocks at the start and end of a `BlockSlice`.
///
/// We only tag B(A+)B sequences, so we need to return this information out-of-band.
#[derive(Clone, Copy, Debug)]
pub struct BlocksAtBoundary {
    pub leading_a_blocks: usize,
    pub trailing_a_blocks: usize,
}

pub struct RollMovementImitationResult<'a, T> {
    pub mvmt: MovementImitationResult<'a, T>,

    /// The final position of the middle block (`mid_blk`) after it has been rolled.
    pub first_b_blk: usize,

    pub trailing_a_blocks: usize,
}

impl<'a, T> RollMovementImitationResult<'a, T> {
    #[debug_requires(blk < self.mvmt.len())]
    pub fn is_b_block(&self, blk: usize, cmp: &mut impl FnMut(&T, &T) -> Ordering) -> bool {
        cmp(&self.mvmt[blk], &self.mvmt[self.first_b_blk]).is_ge()
    }

    pub fn first_b_blk_value(&self) -> &T {
        &self.mvmt[self.first_b_blk]
    }
}

/// Swaps blocks `a` and `b`, reflecting the change in the movement imitation buffer.
fn swap_blocks<T>(
    mut blocks: Blocks<&mut [T]>,
    mvmt: &mut MovementImitationBuffer<'_, T>,
    a: usize,
    b: usize,
) {
    blocks.swap_blocks(a, b);
    mvmt.0.swap(a, b);
}

fn sort_blocks<T>(
    blocks: Blocks<&mut [T]>,
    mvmt: &mut MovementImitationBuffer<'_, T>,
    range: Range<usize>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    if range.is_empty() {
        return;
    }

    for i in range.start..range.end - 1 {
        let min_blk = mvmt[i..]
            .iter()
            .enumerate()
            .min_by(|a, b| cmp(a.1, b.1))
            .map(|x| x.0 + i)
            .unwrap();

        swap_blocks(rbrw!(blocks), mvmt, i, min_blk);
    }
}

/// Finds the index of the minimum block in some subslice of the movement imitation buffer.
///
/// If the subslice is empty, this function returns `usize::MAX`.
fn min_block<T>(
    mvmt: &MovementImitationBuffer<'_, T>,
    range: Range<usize>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> usize {
    let start = range.start;
    mvmt[range]
        .iter()
        .enumerate()
        .min_by(|a, b| cmp(a.1, b.1))
        .map_or(usize::MAX, |(i, _)| i + start)
}

/// Sorts blocks according to their tails.
///
/// Returns the number of A blocks at the end of the sorted sequence of blocks.
#[test_requires(is_sorted_by(&blocks.seq[..mid_blk*blocks.blk_len], cmp))]
#[test_requires(is_sorted_by(&blocks.seq[mid_blk*blocks.blk_len..], cmp))]
#[test_requires(is_sorted_by(&mvmt[..], cmp))]
#[debug_requires(blocks.is_exact())]
#[debug_requires(mvmt.len() >= blocks.num_blocks())]
pub fn roll_blocks_and_tag<T>(
    blocks: Blocks<&mut [T]>,
    mid_blk: usize,
    mut mvmt: MovementImitationBuffer<'_, T>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> BlocksAtBoundary {
    let num_blocks = blocks.num_blocks();

    // The rest of the function assumes that we have at least one block in both A and B,
    //
    // FIXME: This never occurs when called from `sort`, only in unit tests. It should probably
    // be a precondition instead.
    if mid_blk == 0 || mid_blk == num_blocks {
        return BlocksAtBoundary {
            leading_a_blocks: mid_blk,
            trailing_a_blocks: 0,
        };
    }

    let mut a_blk = 0;
    let mid_blk_tail = blocks.tail(mid_blk);

    // Phase 1: Skip over any A blocks less than or equal to the first B block. These blocks are
    // already in the correct place.
    while cmp(blocks.tail(a_blk), mid_blk_tail).is_le() {
        a_blk += 1;

        // If every A block is less than or equal to the first B block, the blocks are already
        // in sorted order. And we don't need to tag anything.
        if a_blk == mid_blk {
            return BlocksAtBoundary {
                leading_a_blocks: a_blk,
                trailing_a_blocks: 0,
            };
        }
    }

    let preceding_a_blocks = a_blk;
    let mut b_blk = mid_blk + 1;

    // We cannot enter phase 2 with zero remaining B blocks, so if we only have a single B block,
    // simply rotate it into place instead of swapping it.
    //
    // FIXME: This can only happen if there is exactly one B block, in which case it is faster to
    // do a quadratic merge. Instead, we could require at least two B blocks as a precondition.
    if b_blk == num_blocks {
        blocks.seq[a_blk * blocks.blk_len..].rotate_right(blocks.blk_len);
        return BlocksAtBoundary {
            leading_a_blocks: preceding_a_blocks,
            trailing_a_blocks: mid_blk - a_blk,
        };
    }

    // Once we find the first A block that is not in the correct order, swap it into place
    swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, mid_blk);

    let mut last_merged_b_blk = a_blk;
    let mut min_a_blk = mid_blk;
    let mut merged_a_blk_since_last_b_blk = false;

    a_blk += 1;

    // Phase 2: Newly merged blocks are placed *before* the scrambled A blocks.
    // In this phase, the list is divided into four sections like so:
    //
    //      |--merged--|----A----|--A(scr)--|---B---|
    //      ^          ^         ^          ^       ^
    //    start      a_blk    mid_blk     b_blk    end
    //
    //
    // - "merged" are A and B blocks that are in sorted order at the start of the list.
    // - "A" are the A blocks that remain in sorted order.
    // - "A (scrambled)" are A blocks that have been swapped with B blocks and so are no longer in
    //   sorted order. All blocks in this list are less than every block in the "A" list.
    // - "B" are the B blocks. They are always in sorted order.
    //
    // At each step, we compare the minimum block in "A (scrambled)" with the first (and therefore
    // minimum) block in "B".
    while a_blk < mid_blk {
        if cmp(blocks.tail(min_a_blk), blocks.tail(b_blk)).is_le() {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, min_a_blk);
            a_blk += 1;
            merged_a_blk_since_last_b_blk = true;

            // Find the new minimum A block in the scrambled list
            //
            // FIXME: We do one more comparison than necessary here, because we know that the A
            // block that was newly swapped into the scrambled list is greater than all other
            // blocks in the scrambled list.
            let scrambled_a_blocks = mid_blk..b_blk;
            min_a_blk = min_block(&mut mvmt, scrambled_a_blocks, cmp);
        } else {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, b_blk);

            // If this is the first B block following a run of A blocks, mark the transition by
            // swapping the head of what was formerly the most recently merged B block with the
            // tail of the newly merged B block.
            if merged_a_blk_since_last_b_blk {
                tag_b_blocks(rbrw!(blocks), last_merged_b_blk, a_blk);
                merged_a_blk_since_last_b_blk = false;
            }

            last_merged_b_blk = a_blk;
            a_blk += 1;
            b_blk += 1;

            // If we run out of B blocks, selection sort the remaining A blocks, using
            // the movement imitation buffer to preserve stability.
            if b_blk == num_blocks {
                sort_blocks(rbrw!(blocks), &mut mvmt, a_blk..b_blk, cmp);
                return BlocksAtBoundary {
                    leading_a_blocks: preceding_a_blocks,
                    trailing_a_blocks: b_blk - a_blk,
                };
            }
        }
    }

    // `b_blk` advanced one beyond `mid_blk` during Phase 1, so it is always strictly greater at
    // this point.
    debug_assert!(a_blk < b_blk);

    // Phase 3: Newly merged elements are placed *inside* the scrambled elements. In this phase,
    // the sorted "A" blocks have been exhausted, leaving only scrambled ones ("A (scrambled)"):
    //
    //
    //      |--merged--|--A(scr)--|---B---|
    //      ^          ^          ^       ^
    //    start      a_blk     b_blk     end
    //
    // It is now possible for the minimum element in the scrambled A blocks to be displaced
    // when we select a B block for merging. When this happens, we need to update the saved index
    // for the minimum A block. Besides that and the fact that the scrambled A blocks no longer begin at
    // `mid_blk` this phase is otherwise the same as the previous one.
    loop {
        if cmp(blocks.tail(min_a_blk), blocks.tail(b_blk)).is_le() {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, min_a_blk);
            a_blk += 1;
            merged_a_blk_since_last_b_blk = true;

            // If we exhaust the last A block, the remaining blocks are B blocks in sorted order at
            // the end of the array. Once we mark the start of a B block series, we are done.
            if a_blk == b_blk {
                tag_b_blocks(rbrw!(blocks), last_merged_b_blk, a_blk);
                break BlocksAtBoundary {
                    leading_a_blocks: preceding_a_blocks,
                    trailing_a_blocks: 0,
                };
            }

            let scrambled_a_blocks = a_blk..b_blk;
            min_a_blk = min_block(&mut mvmt, scrambled_a_blocks, cmp);
        } else {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, b_blk);

            // If the minimum A block is the one we're swapping with, update the pointer to the
            // minimum A block.
            if a_blk == min_a_blk {
                min_a_blk = b_blk;
            }

            if merged_a_blk_since_last_b_blk {
                tag_b_blocks(rbrw!(blocks), last_merged_b_blk, a_blk);
                merged_a_blk_since_last_b_blk = false;
            }

            last_merged_b_blk = a_blk;
            a_blk += 1;
            b_blk += 1;

            if b_blk == num_blocks {
                sort_blocks(rbrw!(blocks), &mut mvmt, a_blk..b_blk, cmp);
                break BlocksAtBoundary {
                    leading_a_blocks: preceding_a_blocks,
                    trailing_a_blocks: b_blk - a_blk,
                };
            }
        }
    }
}

/// Roll blocks without tagging them.
///
/// Their original position is recorded in the movement imitation buffer.
#[test_requires(is_sorted_by(&blocks.seq[..mid_blk*blocks.blk_len], cmp))]
#[test_requires(is_sorted_by(&blocks.seq[mid_blk*blocks.blk_len..], cmp))]
#[test_requires(is_sorted_by(&mvmt[..], cmp))]
#[debug_requires(blocks.is_exact())]
#[debug_requires(mvmt.len() >= blocks.num_blocks())]
pub fn roll_blocks_no_tag<'a, T>(
    blocks: Blocks<&mut [T]>,
    mid_blk: usize,
    mut mvmt: MovementImitationBuffer<'a, T>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> RollMovementImitationResult<'a, T> {
    let num_blocks = blocks.num_blocks();

    // The rest of the function assumes that we have at least one block in both A and B.
    //
    // FIXME: Make this a precondition instead. We should be doing a quadratic merge when one
    // series is smaller than the block size.
    if mid_blk == 0 || mid_blk == num_blocks {
        return RollMovementImitationResult {
            mvmt: mvmt.freeze(),
            first_b_blk: mid_blk,
            trailing_a_blocks: mid_blk,
        };
    }

    let mut a_blk = 0;

    // Phase 1
    let mid_blk_tail = blocks.tail(mid_blk);
    while cmp(blocks.tail(a_blk), mid_blk_tail).is_le() {
        a_blk += 1;
        if a_blk == mid_blk {
            return RollMovementImitationResult {
                mvmt: mvmt.freeze(),
                first_b_blk: mid_blk,
                trailing_a_blocks: 0,
            };
        }
    }

    let mut b_blk = mid_blk + 1;

    if b_blk == num_blocks {
        blocks.seq[a_blk * blocks.blk_len..].rotate_right(blocks.blk_len);
        mvmt.0[a_blk..].rotate_right(1);

        return RollMovementImitationResult {
            mvmt: mvmt.freeze(),
            first_b_blk: a_blk,
            trailing_a_blocks: mid_blk - a_blk,
        };
    }

    let first_b_blk_final = a_blk;
    swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, mid_blk);

    a_blk += 1;
    let mut min_a_blk = mid_blk;

    // Phase 2
    while a_blk < mid_blk {
        if cmp(blocks.tail(min_a_blk), blocks.tail(b_blk)).is_le() {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, min_a_blk);
            a_blk += 1;

            let scrambled_a_blocks = mid_blk..b_blk;
            min_a_blk = min_block(&mvmt, scrambled_a_blocks, cmp);
        } else {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, b_blk);
            a_blk += 1;
            b_blk += 1;

            if b_blk == num_blocks {
                sort_blocks(rbrw!(blocks), &mut mvmt, a_blk..b_blk, cmp);
                return RollMovementImitationResult {
                    mvmt: mvmt.freeze(),
                    first_b_blk: first_b_blk_final,
                    trailing_a_blocks: b_blk - a_blk,
                };
            }
        }
    }

    debug_assert!(a_blk < b_blk);

    // Phase 3
    loop {
        if cmp(blocks.tail(min_a_blk), blocks.tail(b_blk)).is_le() {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, min_a_blk);
            a_blk += 1;

            if a_blk == b_blk {
                break RollMovementImitationResult {
                    mvmt: mvmt.freeze(),
                    first_b_blk: first_b_blk_final,
                    trailing_a_blocks: 0,
                };
            }

            let scrambled_a_blocks = a_blk..b_blk;
            min_a_blk = min_block(&mut mvmt, scrambled_a_blocks, cmp);
        } else {
            swap_blocks(rbrw!(blocks), &mut mvmt, a_blk, b_blk);

            if a_blk == min_a_blk {
                min_a_blk = b_blk;
            }

            a_blk += 1;
            b_blk += 1;

            if b_blk == num_blocks {
                sort_blocks(blocks, &mut mvmt, a_blk..b_blk, cmp);
                break RollMovementImitationResult {
                    mvmt: mvmt.freeze(),
                    first_b_blk: first_b_blk_final,
                    trailing_a_blocks: b_blk - a_blk,
                };
            }
        }
    }
}

#[debug_requires(blk < blocks.num_blocks())]
#[debug_requires(blk != 0)]
pub fn is_first_b_block_in_series<T>(
    blocks: Blocks<&[T]>,
    blk: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> bool {
    // If the tail of this block is less than the tail of the preceding block, this is
    // the first block in this series of B blocks.
    cmp(blocks.tail(blk), blocks.tail(blk - 1)).is_lt()
}

#[debug_requires(blk < blocks.num_blocks())]
pub fn is_last_b_block_in_series<T>(
    blocks: Blocks<&[T]>,
    blk: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> bool {
    // If the head of this block is greater than its successor in the same block, this
    // is the last block in this series of B blocks.
    cmp(blocks.head(blk), &blocks.block(blk)[1]).is_gt()
}

#[debug_requires(final_blk_in_prev_series < blocks.num_blocks())]
#[debug_requires(first_blk_in_next_series < blocks.num_blocks())]
#[debug_requires(final_blk_in_prev_series + 1 < first_blk_in_next_series, "At least one A block between this and the last B block")]
pub fn tag_b_blocks<T>(
    mut blocks: Blocks<&mut [T]>,
    final_blk_in_prev_series: usize,
    first_blk_in_next_series: usize,
) {
    blocks.swap_head_tail(final_blk_in_prev_series, first_blk_in_next_series);
}

// B block tagging is its own inverse (involution).
pub use tag_b_blocks as untag_b_blocks;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{Blk, Blocks, SplitBlocks};
    use crate::test_utils::{cmp_ignore_idx, sorted, vec_blocks_exact_split, KeyAndIndex};
    use crate::Split;
    use proptest::prelude::*;
    use std::convert::TryFrom;
    use std::iter;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum BlockKind {
        A,
        B,
    }

    #[test]
    fn block_kind_ordering() {
        assert!(BlockKind::A < BlockKind::B);
    }

    #[rustfmt::skip]
    fn roll_blocks_slow<T>(
        mut blocks: SplitBlocks<&mut [T]>,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
    ) -> Vec<BlockKind> {
        let Blk(mid_blk) = blocks.at;

        let num_blocks = blocks.seq.num_blocks();
        if num_blocks == 0 {
            return vec![];
        }

        let mut idxs: Vec<usize> = (0..num_blocks).collect();

        for i in 0..num_blocks - 1 {
            let min_block = blocks
                .seq
                .tails_enumerated()
                .skip(i)
                .min_by(|a, b| cmp(a.1, b.1).then_with(|| idxs[a.0].cmp(&idxs[b.0])))
                .unwrap()
                .0;

            blocks.seq.swap_blocks(i, min_block);
            idxs.swap(i, min_block);
        }

        idxs.into_iter()
            .map(|i| if i < mid_blk { BlockKind::A } else { BlockKind::B })
            .collect()
    }

    #[rustfmt::skip]
    fn extract_block_tags<T>(
        blocks: Blocks<&mut [T]>,
        acnt: BlocksAtBoundary,
        cmp: &mut impl FnMut(&T, &T) -> Ordering,
    ) -> Vec<BlockKind> {
        let num_blocks = blocks.num_blocks();
        let mut ret = Vec::with_capacity(num_blocks);
        ret.extend(iter::repeat(BlockKind::A).take(acnt.leading_a_blocks));

        let mut prev_b_blk = None;
        let mut in_b_block_series = true;
        for blk in acnt.leading_a_blocks..(num_blocks - acnt.trailing_a_blocks) {
            if !in_b_block_series {
                if is_first_b_block_in_series(rbrw!(blocks).as_shared(), blk, cmp) {
                    untag_b_blocks(rbrw!(blocks), prev_b_blk.unwrap(), blk);
                    prev_b_blk = None;
                    in_b_block_series = true;
                }
            }

            ret.push(if in_b_block_series { BlockKind::B } else { BlockKind::A });

            if in_b_block_series {
                if is_last_b_block_in_series(rbrw!(blocks).as_shared(), blk, cmp) {
                    prev_b_blk = Some(blk);
                    in_b_block_series = false;
                }
            }
        }

        ret.extend(iter::repeat(BlockKind::A).take(acnt.trailing_a_blocks));
        ret
    }

    fn test_roll_blocks(input: SplitBlocks<Vec<u8>>) -> Result<(), TestCaseError> {
        let input = input.map_seq(|chnk| chnk.map_seq(KeyAndIndex::map_vec));
        let cmp = &mut cmp_ignore_idx;

        // Expected

        let mut expected = input.clone();
        let expected_block_kinds = roll_blocks_slow(
            Split::new(
                Blocks::new(&mut expected.seq.seq, expected.seq.blk_len),
                expected.at,
            ),
            cmp,
        );

        // Actual

        let Blk(mid_blk) = input.at;
        let mut actual = input;

        let mut mvmt: Vec<KeyAndIndex<_>> = (0..actual.seq.num_blocks())
            .map(|x| KeyAndIndex::without_index(u8::try_from(x).unwrap()))
            .collect();

        let mvmt = MovementImitationBuffer::new(&mut mvmt[..], cmp);
        let rslt = super::roll_blocks_and_tag(actual.seq.as_mut(), mid_blk, mvmt, cmp);
        let actual_block_kinds = extract_block_tags(actual.seq.as_mut(), rslt, cmp);

        prop_assert_eq!(actual_block_kinds, expected_block_kinds);
        prop_assert_eq!(actual.seq.seq, expected.seq.seq);
        Ok(())
    }

    fn test_roll_blocks_no_tag(input: SplitBlocks<Vec<u8>>) -> Result<(), TestCaseError> {
        let input = input.map_seq(|chnk| chnk.map_seq(KeyAndIndex::map_vec));
        let cmp = &mut cmp_ignore_idx;

        // Expected

        let mut expected = input.clone();
        let expected_block_kinds = roll_blocks_slow(
            Split::new(
                Blocks::new(&mut expected.seq.seq, expected.seq.blk_len),
                expected.at,
            ),
            cmp,
        );

        // Actual

        let Blk(mid_blk) = input.at;
        let mut actual = input;

        let mut mvmt: Vec<KeyAndIndex<_>> = (0..actual.seq.num_blocks())
            .map(|x| KeyAndIndex::without_index(u8::try_from(x).unwrap()))
            .collect();

        let mvmt = MovementImitationBuffer::new(&mut mvmt, cmp);
        let rslt = super::roll_blocks_no_tag(actual.seq.as_mut(), mid_blk, mvmt, cmp);

        // FIXME: Ban the case where we have no B blocks.
        let actual_block_kinds: Vec<_> = if rslt.first_b_blk == rslt.mvmt.len() {
            vec![BlockKind::A; rslt.first_b_blk]
        } else {
            (0..rslt.mvmt.len())
                .map(|blk| {
                    if rslt.is_b_block(blk, cmp) {
                        BlockKind::B
                    } else {
                        BlockKind::A
                    }
                })
                .collect()
        };

        prop_assert_eq!(actual_block_kinds, expected_block_kinds);
        prop_assert_eq!(actual.seq.seq, expected.seq.seq);
        Ok(())
    }

    #[test]
    fn roll_blocks_oneshot() {
        let input: Vec<u8> = vec![
            23, 40, 40, 41, 42, 48, 5, 5, 8, 10, 10, 11, 14, 16, 18, 25, 30, 36,
        ];
        let input = Blocks::new(input, 2);
        let input = Split::new(input, Blk(3));

        prop_unwrap!(test_roll_blocks(input));
    }

    #[test]
    fn roll_blocks_no_tag_oneshot() {
        let input: Vec<u8> = vec![
            23, 40, 40, 41, 42, 48, 5, 5, 8, 10, 10, 11, 14, 16, 18, 25, 30, 36,
        ];
        let input = Blocks::new(input, 2);
        let input = Split::new(input, Blk(3));

        prop_unwrap!(test_roll_blocks_no_tag(input));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn roll_blocks(x in sorted(vec_blocks_exact_split(2..24, 2..4, 0u8..30))) {
            test_roll_blocks(x)?;
        }

        #[test]
        fn roll_blocks_no_tag(x in sorted(vec_blocks_exact_split(2..24, 2..4, 0u8..30))) {
            test_roll_blocks_no_tag(x)?;
        }
    }
}
