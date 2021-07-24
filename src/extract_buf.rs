use std::cmp::Ordering;

/// Finds the first `k` distinct elements in `slice` and rotates them to the start, returning the
/// number of elements found.
///
/// If `slice` has fewer than `k` distinct elements, all of them will be rotated to the start, and
/// the return value will be less than `k`. Otherwise, the return value is equal to `k`.
pub fn extract_k_distinct<T>(
    slice: &mut [T],
    k: usize,
    cmp: &mut impl FnMut(&T, &T) -> Ordering,
) -> usize {
    if slice.is_empty() || k == 0 {
        return 0;
    }

    let k_target = k;

    // The first element of the list is always new.
    let mut k = 1;

    // `s` advances through the `slice` as we iterate. It consists of two regions:
    //  - `s[..k]` are the distinct elements we have found so far.
    //  - `s[k..]` are the elements we have yet to look at.
    //
    //  We must store it as a single slice because we need to rotate across the midpoint.
    let mut s = &mut slice[..];

    while k < k_target && k < s.len() {
        let (distinct, other) = s.split_at(k);

        let mut prev_el = None;
        let find_next_uniq_pos = |(i, el)| {
            if prev_el.map_or(false, |prev_el| cmp(el, prev_el).is_eq()) {
                return None;
            }

            prev_el = Some(el);

            // `Err` indicates that we failed to find the element.
            distinct.binary_search_by(|u| cmp(u, el))
                .err()
                .map(|uidx| (i, uidx))
        };

        let next_uniq_pos = other.iter().enumerate().find_map(find_next_uniq_pos);

        let (
            el_idx,       // The index of the next distinct element in `s[k..]`.
            distinct_idx, // The index that element will occupy in `uniq` when sorted.
        ) = match next_uniq_pos {
            None => break,
            Some(x) => x,
        };

        // Minimize the rotation distance by checking whether the final, sorted position of `el` is
        // closer to the start or end of `uniq` and adjusting the initial rotation accordingly.
        if distinct_idx >= k / 2 {
            // Rotate previously found distinct elements *before* the newly found one.
            // |----------)
            // [uuuu uu.. e...] -> [..uu uuuu e...]
            s[..el_idx + k].rotate_left(k);

            // Advance `s` past any duplicate elements.
            s = &mut s[el_idx..];
            k += 1;

            // Rotate the newly found distinct element into sorted order.
            //       |--)
            // [uuuu uue.] -> [uuuu euu.]
            if distinct_idx != k {
                s[distinct_idx..k].rotate_right(1);
            }
        } else {
            // Rotate previously found distinct elements *after* the newly found one.
            // |------- ---)
            // [uuuu uu.. e...] -> [..eu uuuu u...]
            s[..el_idx + k + 1].rotate_left(k);

            // See above.
            s = &mut s[el_idx..];
            k += 1;

            //  |--)
            // [euuu uuu.] -> [uueu uuu.]
            if distinct_idx != 0 {
                s[..distinct_idx + 1].rotate_left(1);
            }
        }
    }

    // Rotate the entire slice to put the distinct elements at the start.
    // |-----------)
    // [.... uuuu u...]  -> [uuuu u... ....]
    let leftover = s.len() - k;
    let total = slice.len();
    slice[..total - leftover].rotate_right(k);

    k
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    use crate::test_utils::{cmp_ignore_idx, KeyAndIndex};

    #[test]
    fn extract_k_distinct_oneshot() {
        let input = vec![22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 22];
        let k = 5;

        prop_unwrap!(test_extract_k_distinct(input, k));
    }

    fn test_extract_k_distinct(input: Vec<u8>, k: usize) -> Result<(), TestCaseError> {
        let cmp = &mut cmp_ignore_idx;
        let mut input = KeyAndIndex::map_vec(input);

        let (mut expect_uniq, expect_rem) = extract_k_distinct_slow(&input, k);
        let actual_num_uniq = super::extract_k_distinct(&mut input, k, cmp);

        prop_assert_eq!(actual_num_uniq, expect_uniq.len());

        let (actual_uniq, actual_rem) = input.split_at_mut(actual_num_uniq);
        let actual_rem: &_ = actual_rem;

        // The relative order of distinct elements is not important.
        expect_uniq.sort_unstable();
        actual_uniq.sort_unstable();

        prop_assert_eq!(actual_uniq, expect_uniq);
        prop_assert_eq!(actual_rem, expect_rem);
        Ok(())
    }

    fn extract_k_distinct_slow(
        s: &[KeyAndIndex<u8>],
        k: usize,
    ) -> (Vec<KeyAndIndex<u8>>, Vec<KeyAndIndex<u8>>) {
        let mut uniq_keys = BTreeSet::new();

        if k == 0 {
            return (vec![], s.to_vec());
        }

        let mut rem = vec![];
        let mut uniq = vec![];

        let mut iter = s.iter();
        while let Some(&el) = iter.next() {
            let is_uniq = uniq_keys.insert(el.key);
            if is_uniq {
                uniq.push(el);
            } else {
                rem.push(el);
            }

            if uniq_keys.len() == k {
                break;
            }
        }

        rem.extend(iter);
        (uniq, rem)
    }

    proptest! {
        #[test]
        fn extract_k_distinct(input in proptest::collection::vec(0u8..39, 0..100), k in 0usize..30) {
            test_extract_k_distinct(input, k)?;
        }
    }
}
