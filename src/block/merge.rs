//! Merge runs whose size is larger than the internal buffer.

use super::roll::{BlocksAtBoundary, RollMovementImitationResult};
use super::Blocks;
use crate::merge_buf::ContiguousBlockPairAndLeftBuffer;
use crate::merge_no_buf::merge_in_place_quadratic_partial;
use crate::search::linear_search;
use crate::{Either, Split};
use contracts::*;
use std::cmp::Ordering;
use std::ops::Range;

#[cfg(test)]
use crate::test_utils::is_sorted_by;

#[debug_requires(leading_a_blocks + trailing_a_blocks < buf_and_blocks.num_blocks())]
pub fn merge_rolled_blocks_tagged<T>(
    buf_and_blocks: Blocks<&mut [T]>,
    BlocksAtBoundary {
        leading_a_blocks,
        trailing_a_blocks,
    }: BlocksAtBoundary,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    let blk_len = buf_and_blocks.blk_len;
    let final_a_block_series_start = buf_and_blocks.num_blocks() - trailing_a_blocks;

    // N.B. This needs to be set to `block_size`, because if have no preceding A blocks,
    // we need to include the *entire* first B block in the initial merge.
    let mut rem_from_last_merge = blk_len;

    let mut ablk = 1; // Add one to account for the aux buffer.
    let mut bblk = ablk + leading_a_blocks;

    // Find the end of the first B block series and the start of the next one. We must do this now,
    // since if the B block series is one block long, the head of that block will be swapped with a
    // later element to mark it as the end of a B block series. We cannot merge such a block before
    // untagging it.
    let mut b_block_curr_end_and_next_start = find_and_untag_b_block_end_start_pair(
        rbrw!(buf_and_blocks),
        bblk..final_a_block_series_start,
        cmp,
    );

    if leading_a_blocks != 0 {
        // Merge AB₀
        rem_from_last_merge = merge_block_series(
            rbrw!(buf_and_blocks),
            ablk,
            bblk,
            rem_from_last_merge,
            cmp,
            Either::Left(()),
            merge_series_with_aux_buf,
        )
        .expect_right("B block should have tail >= than anything in A blocks");
    }

    // Invariant: We are merging a BAB block series where the last B block in the first
    // series and the first B block of the second series have already been untagged.
    while let Some((b_block_curr_end, b_block_next_start)) = b_block_curr_end_and_next_start {
        // Merge BA₀
        ablk = b_block_curr_end + 1;
        rem_from_last_merge = merge_block_series(
            rbrw!(buf_and_blocks),
            bblk,
            ablk,
            rem_from_last_merge,
            cmp,
            Either::Right(()),
            merge_series_with_aux_buf,
        )
        .expect_right("B block should have tail >= than anything in A blocks");

        // Once again, we have to do this before the AB₀ merge in case the next B block series has
        // only a single block.
        b_block_curr_end_and_next_start = find_and_untag_b_block_end_start_pair(
            rbrw!(buf_and_blocks),
            b_block_next_start..final_a_block_series_start,
            cmp,
        );

        // Merge AB₀
        bblk = b_block_next_start;
        rem_from_last_merge = merge_block_series(
            rbrw!(buf_and_blocks),
            ablk,
            bblk,
            rem_from_last_merge,
            cmp,
            Either::Left(()),
            merge_series_with_aux_buf,
        )
        .expect_right("B block should have tail >= than anything in A blocks");
    }

    let buf_start = if trailing_a_blocks != 0 {
        // Merge BA₀
        ablk = final_a_block_series_start;
        let buf_end = merge_final_a_block_series(
            rbrw!(buf_and_blocks),
            rem_from_last_merge,
            bblk,
            ablk,
            cmp,
            merge_series_with_aux_buf,
        );

        buf_end - blk_len
    } else {
        (bblk * blk_len) - rem_from_last_merge
    };

    // Rotate auxiliary buffer to the end.
    buf_and_blocks.seq[buf_start..].rotate_left(blk_len);
}

#[ensures(ret.map_or(true, |(end, next_start)| end + 1 < next_start))]
fn find_and_untag_b_block_end_start_pair<T>(
    buf_and_blocks: Blocks<&mut [T]>,
    block_range: Range<usize>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> Option<(usize, usize)> {
    use super::roll::{is_first_b_block_in_series, is_last_b_block_in_series, untag_b_blocks};

    let mut block_range = block_range;

    block_range
        .find(|&blk| is_last_b_block_in_series(rbrw!(buf_and_blocks).as_shared(), blk, cmp))
        .map(|series_end| {
            let next_start = (series_end + 1..block_range.end)
                .find(|&blk| {
                    is_first_b_block_in_series(rbrw!(buf_and_blocks).as_shared(), blk, cmp)
                })
                .expect("Unmatched B block tag");

            untag_b_blocks(rbrw!(buf_and_blocks), series_end, next_start);
            (series_end, next_start)
        })
}

#[debug_requires(mvmt.mvmt.len() >= blocks.num_blocks())]
#[debug_requires(mvmt.first_b_blk < blocks.num_blocks(), "Cannot merge with zero B blocks")]
#[test_requires(blocks.iter().all(|s| is_sorted_by(s, cmp)))]
pub fn merge_rolled_blocks_untagged<T>(
    blocks: Blocks<&mut [T]>,
    mvmt: RollMovementImitationResult<'_, T>,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) {
    let num_blks = blocks.num_blocks();
    let block_size = blocks.blk_len;

    // The number of elements that were not fully merged from the first block in the right-sries.
    let mut rem_from_last_merge = block_size;

    let mut lblk = 0;
    let mut a_blk_side = if mvmt.is_b_block(lblk, cmp) {
        Either::Right(())
    } else {
        Either::Left(())
    };

    let RollMovementImitationResult {
        mvmt,
        first_b_blk,
        trailing_a_blocks,
    } = mvmt;

    // We use the movement imitation buffer to identify the kind of a block, so we only need to
    // search within a subset.
    let mvmt = mvmt.subslice(0..num_blks);

    let mvmt_mid_blk_val = &mvmt[first_b_blk];
    let final_a_blk_series_start = num_blks - trailing_a_blocks;

    loop {
        // `rblk` is the next block from the opposite half of the original sorted chunk (A or B).
        let rblk = (lblk + 1)
            + linear_search(&mvmt[lblk + 1..], |x| {
                let x_is_a_blk = cmp(x, mvmt_mid_blk_val).is_lt();
                let lblk_is_a_blk = a_blk_side.is_left();
                x_is_a_blk == lblk_is_a_blk
            });

        // Exit prior to merging the first block of the final A block series.
        //
        // If the final series is a B block one, `final_a_blk_series_start` is equal to `num_blks`
        // and the merge is complete.
        debug_assert!(rblk <= final_a_blk_series_start);
        if rblk == final_a_blk_series_start {
            break;
        }

        rem_from_last_merge = merge_block_series(
            rbrw!(blocks),
            lblk,
            rblk,
            rem_from_last_merge,
            cmp,
            a_blk_side,
            merge_series_quadratic,
        )
        .expect_right("Right series tail must be greater than left series");

        lblk = rblk;
        a_blk_side = a_blk_side.flip();
    }

    // If our final full block is a B block, we're done (even if there's a partial B block).
    //
    // Otherwise, we need to merge the final A block series.
    if final_a_blk_series_start != num_blks {
        merge_final_a_block_series(
            blocks,
            rem_from_last_merge,
            lblk,
            final_a_blk_series_start,
            cmp,
            merge_series_quadratic,
        );
    }
}

/// Performs the merge of the final A block series, handling a partial B block if one exists.
///
/// Returns the position of the remaining elements from the last partial merge. This is also the
/// end of the auxiliary buffer if one was used.
#[debug_requires(final_a_blk_series_start < blocks.num_blocks())]
#[debug_requires(final_b_blk_series_start < final_a_blk_series_start)]
fn merge_final_a_block_series<T, C>(
    blocks: Blocks<&mut [T]>,
    mut rem_from_last_merge: usize,
    final_b_blk_series_start: usize,
    final_a_blk_series_start: usize,
    cmp: &mut C,
    mut merge: impl FnMut(Blocks<&mut [T]>, usize, usize, usize, &mut C, Either) -> Either<usize>,
) -> usize
where
    C: FnMut(&T, &T) -> Ordering,
{
    let num_blks = blocks.num_blocks();
    let block_size = blocks.blk_len;

    let partial_blk_size = blocks.seq.len() % block_size;

    let partial_blk_pos = if partial_blk_size > 0 {
        let partial_blk_tail = blocks.seq.last().unwrap();

        // Find the first A-block with a tail strictly greater than the partial block's. The
        // partial block belongs immediately before this one.
        let partial_blk_pos = blocks
            .tails_enumerated()
            .skip(final_a_blk_series_start)
            .find(|(_, tail)| cmp(tail, partial_blk_tail).is_gt())
            .map_or(num_blks, |(i, _)| i);

        // Rotate the partial block into place...
        blocks.seq[partial_blk_pos * blocks.blk_len..].rotate_right(partial_blk_size);
        partial_blk_pos
    } else {
        // When we don't have a partial block, treat it like case 1 below.
        final_a_blk_series_start
    };

    // At this point there are three possibilities:
    //
    // 1) The partial B block (b) belongs before any A block in the final series
    //    (`partial_blk_pos == final_a_blk_series_start`)
    //            BBbAAAAA
    // 2) The partial B block is in the middle of the final A block series.
    //            BBAAAbAA
    // 3) The partial B block is at the end of the final A block series
    //    (`partial_blk_pos == num_blks`)
    //            BBAAAAAb

    // For case 1, account for the extra elements at the end of the final B block series.
    let final_b_blk_series_extra = if partial_blk_pos == final_a_blk_series_start {
        partial_blk_size
    } else {
        0
    };

    // Merge the final B block series with the first block of the final A block series.
    //   BBB(b)A
    let start = (final_b_blk_series_start + 1) * block_size - rem_from_last_merge;
    let mid = final_a_blk_series_start * block_size + final_b_blk_series_extra;
    let end = (final_a_blk_series_start + 1) * block_size + final_b_blk_series_extra;

    rem_from_last_merge = merge(rbrw!(blocks), start, mid, end, cmp, Either::Right(()))
        .expect_right("Right series tail must be greater than left series");

    // If we're in case 1, all remaining blocks are A blocks. We're done.
    if partial_blk_pos == final_a_blk_series_start {
        return end - rem_from_last_merge;
    }

    // Otherwise, merge the first part of the final A block series with the partial B block.

    let partial_blk_start = partial_blk_pos * block_size;
    let partial_blk_end = partial_blk_start + partial_blk_size;

    let start = (final_a_blk_series_start + 1) * block_size - rem_from_last_merge;
    let mid = partial_blk_pos * block_size;
    let end = partial_blk_end;

    // AAAAb
    rem_from_last_merge = merge(rbrw!(blocks), start, mid, end, cmp, Either::Left(()))
        .expect_right("Right series tail must be greater than left series");

    // If we're in case 3, there are no A blocks after the partial block. We're done.
    if partial_blk_pos == num_blks {
        return end - rem_from_last_merge;
    }

    let start = partial_blk_end - rem_from_last_merge;
    let mid = partial_blk_end;
    let end = partial_blk_end + block_size;

    // bA
    rem_from_last_merge = merge(rbrw!(blocks), start, mid, end, cmp, Either::Right(()))
        .expect_right("Right series tail must be greater than left series");

    end - rem_from_last_merge
}

/// Merges the block series starting at block `lblk` (less `rem_in_lblk` elements) with the block
/// starting at `rblk` using the given `merge` subroutine.
fn merge_block_series<T, C>(
    blocks: Blocks<&mut [T]>,
    lblk: usize,
    rblk: usize,
    rem_in_lblk: usize,
    cmp: &mut C,
    prefer: Either,
    merge: impl FnOnce(Blocks<&mut [T]>, usize, usize, usize, &mut C, Either) -> Either<usize>,
) -> Either<usize>
where
    C: FnMut(&T, &T) -> Ordering,
{
    let block_size = blocks.blk_len;

    let start = (lblk + 1) * block_size - rem_in_lblk;
    let mid = rblk * block_size;
    let end = (rblk + 1) * block_size;

    merge(blocks, start, mid, end, cmp, prefer)
}

/// Merges the left series with the first block of the right series.
fn merge_series_quadratic<T>(
    blocks: Blocks<&mut [T]>,
    start: usize,
    mid: usize,
    end: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) -> Either<usize> {
    merge_in_place_quadratic_partial(
        Split::new(&mut blocks.seq[start..end], mid - start),
        cmp,
        prefer,
    )
}

fn merge_series_with_aux_buf<T>(
    buf_and_blocks: Blocks<&mut [T]>,
    start: usize,
    mid: usize,
    end: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
    prefer: Either,
) -> Either<usize> {
    let aux_buf_size = buf_and_blocks.blk_len;
    let buf_start = start - aux_buf_size;

    ContiguousBlockPairAndLeftBuffer::new(
        &mut buf_and_blocks.seq[buf_start..end],
        aux_buf_size,
        mid - start,
    )
    .merge_partial(cmp, prefer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::roll::{roll_blocks_and_tag, roll_blocks_no_tag, MovementImitationBuffer};
    use crate::block::{Blk, SplitBlocks};
    use crate::test_utils::{cmp_ignore_idx, nonempty, sorted, vec_blocks_split, KeyAndIndex};
    use proptest::prelude::*;

    fn test_merge_rolled_blocks(input: SplitBlocks<Vec<i32>>) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input: SplitBlocks<_> = input.map_seq(|split| split.map_seq(KeyAndIndex::map_vec));
        let Blk(mid_blk) = input.at;
        let block_size = input.seq.blk_len;

        // Use an external movement imitation buffer to roll blocks.
        //
        // If we tried to use the dummy elements at the start of `actual`, we would have to ensure
        // that there were enough to hold both the number of blocks *and* all elements in a block.
        let mut mvmt: Vec<_> = (i32::MIN..0)
            .take(input.seq.num_blocks())
            .map(KeyAndIndex::without_index)
            .collect();
        let mvmt = MovementImitationBuffer::new(&mut mvmt, cmp);

        // Expected

        let mut expected = input.seq.seq.clone();
        expected.sort_unstable();

        // Actual

        let aux_buf_size = block_size;
        let aux_elems = (i32::MIN..0).take(aux_buf_size);
        let mut actual: Vec<_> = aux_elems.clone().map(KeyAndIndex::without_index).collect();
        actual.extend(input.seq.seq.into_iter());

        let roll_input_end = actual.len() - (actual.len() % block_size);
        let rslt = roll_blocks_and_tag(
            Blocks::new(&mut actual[aux_buf_size..roll_input_end], block_size),
            mid_blk,
            mvmt,
            cmp,
        );
        super::merge_rolled_blocks_tagged(Blocks::new(&mut actual[..], block_size), rslt, cmp);

        // Compare

        let actual_len = actual.len();
        let (actual_blocks, aux_buf) = actual.split_at_mut(actual_len - aux_buf_size);
        prop_assert_eq!(actual_blocks, &expected);

        aux_buf.sort_unstable();
        prop_assert!(aux_buf.iter().map(|x| x.key).eq(aux_elems));
        Ok(())
    }

    fn test_merge_untagged_blocks(input: SplitBlocks<Vec<i32>>) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let input: SplitBlocks<_> = input.map_seq(|split| split.map_seq(KeyAndIndex::map_vec));
        let Blk(mid_blk) = input.at;

        // Expected

        let mut expected = input.seq.seq.clone();
        expected.sort_unstable();

        let mut actual = input.seq;
        let mut aux_buf: Vec<_> = (i32::MIN..0)
            .take(actual.num_blocks())
            .map(KeyAndIndex::without_index)
            .collect();
        let mvmt = MovementImitationBuffer::new(aux_buf.as_mut(), cmp);

        let roll_input_end = actual.seq.len() - (actual.seq.len() % actual.blk_len);
        let actual_exact = Blocks::new(&mut actual.seq[..roll_input_end], actual.blk_len);
        let rslt = roll_blocks_no_tag(actual_exact, mid_blk, mvmt, cmp);

        super::merge_rolled_blocks_untagged(actual.as_mut(), rslt, cmp);
        prop_assert_eq!(actual.seq, expected);
        Ok(())
    }

    #[test]
    fn merge_rolled_blocks_oneshot() {
        let input = vec![10, 22, 2, 5, 5, 7, 11, 12, 13, 14, 16, 26];
        let input = Blocks::new(input, 2);
        let input = Split::new(input, Blk(1));

        prop_unwrap!(test_merge_rolled_blocks(input));
    }

    #[test]
    fn merge_untagged_blocks_oneshot() {
        let input = vec![10, 22, 2, 5, 5, 7, 11, 12, 13, 14, 16, 26];
        let input = Blocks::new(input, 2);
        let input = Split::new(input, Blk(1));

        prop_unwrap!(test_merge_untagged_blocks(input));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn merge_rolled_blocks(
            input in sorted(nonempty(vec_blocks_split(4..24, 2..8, 0i32..30)))
        ) {
            test_merge_rolled_blocks(input)?;
        }

        #[test]
        fn merge_untagged_blocks(
            input in sorted(nonempty(vec_blocks_split(4..24, 2..8, 0i32..30)))
        ) {
            test_merge_untagged_blocks(input)?;
        }
    }
}
