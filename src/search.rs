//! Various search algorithms (linear, binary, exponential) with a common interface.

use contracts::*;
use std::ops::Range;

/// Returns the smallest index for which `is_before(&s[i])` returns `false`, or `s.len()` if no
/// such index exists.
///
/// `s` must be partitioned such that, for some integer `N`, `is_before` returns `true` for the
/// first `N` elements, then `false` for the remaining ones (e.g. `[TTTT TFFF FFFF]`).
#[debug_ensures(ret <= s.len())]
pub fn binary_search<T>(s: &[T], is_before: impl FnMut(&T) -> bool) -> usize {
    binary_search_range(s, 0..s.len(), is_before)
}

#[debug_requires(range.start <= range.end)]
#[debug_requires(range.end <= s.len())]
#[debug_ensures(ret <= s.len())]
fn binary_search_range<T>(
    s: &[T],
    range: Range<usize>,
    mut is_before: impl FnMut(&T) -> bool,
) -> usize {
    let mut hi = range.end;
    let mut lo = range.start;

    while lo < hi {
        // Compute `lo + hi` / 2 without overflow.
        let mid = lo + (hi - lo) / 2;

        // LLVM fails to optimize away the bounds check, so we need `get_unchecked`.
        let el: &T = unsafe { s.get_unchecked(mid) };
        if is_before(el) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo
}

/// Returns the smallest index for which `is_before(&s[i])` returns `false`, or `s.len()` if no
/// such index exists.
///
/// `s` must be partitioned such that, for some integer `N`, `is_before` returns `true` for the
/// first `N` elements, then `false` for the remaining ones (e.g. `[TTTT TFFF FFFF]`).
#[debug_ensures(ret <= s.len())]
pub fn exp_search<T>(s: &[T], mut is_before: impl FnMut(&T) -> bool) -> usize {
    // Fast path for when the target is at the start of the list.
    match s.first() {
        None => return 0,
        Some(el) if !is_before(el) => return 0,
        _ => {}
    }

    let mut prev_idx = 0usize;
    let mut stride = 1;

    // INVARIANT: `is_before(s[prev_idx])` returned `true`
    loop {
        let (idx, overflow) = prev_idx.overflowing_add(stride);
        if overflow || idx >= s.len() {
            return binary_search_range(s, prev_idx..s.len(), is_before);
        }

        if !is_before(&s[idx]) {
            return binary_search_range(s, prev_idx..idx, is_before);
        }

        stride *= 2;
        prev_idx = idx;
    }
}

#[debug_ensures(ret <= s.len())]
pub fn rexp_search<T>(s: &[T], mut is_before: impl FnMut(&T) -> bool) -> usize {
    // Fast path for when the target is at the end of the list.
    match s.last() {
        None => return 0,
        Some(el) if is_before(el) => return s.len(),
        _ => {}
    }

    let mut prev_idx = s.len() - 1;
    let mut stride = 1;

    // INVARIANT: `prev_idx < s.len()`
    // INVARIANT: `is_before(s[prev_idx])` returned `false`
    loop {
        let (idx, overflow) = prev_idx.overflowing_sub(stride);
        if overflow {
            return binary_search_range(s, 0..prev_idx, is_before);
        }

        if is_before(&s[idx]) {
            return binary_search_range(s, idx + 1..prev_idx, is_before);
        }

        stride *= 2;
        prev_idx = idx;
    }
}

#[debug_ensures(ret <= s.len())]
pub fn linear_search<T>(s: &[T], mut is_before: impl FnMut(&T) -> bool) -> usize {
    s.iter().position(|el| !is_before(el)).unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort_unstable();
        v
    }

    const VALS: std::ops::Range<u8> = 0..30;

    proptest! {
        #[test]
        fn binary_search(arr in proptest::collection::vec(VALS, 0..100).prop_map(sorted)) {
            for targ in VALS {
                let mut cmp = |x: &u8| *x < targ;
                prop_assert_eq!(
                    super::linear_search(&arr, &mut cmp),
                    super::binary_search(&arr, &mut cmp)
                );
            }
        }

        #[test]
        fn exp_search(arr in proptest::collection::vec(VALS, 0..100).prop_map(sorted)) {
            for targ in VALS {
                let mut cmp = |x: &u8| *x < targ;
                prop_assert_eq!(
                    super::linear_search(&arr, &mut cmp),
                    super::exp_search(&arr, &mut cmp)
                );
            }
        }

        #[test]
        fn rexp_search(arr in proptest::collection::vec(VALS, 0..100).prop_map(sorted)) {
            for targ in VALS {
                let mut cmp = |x: &u8| *x < targ;
                prop_assert_eq!(
                    super::linear_search(&arr, &mut cmp),
                    super::rexp_search(&arr, &mut cmp)
                );
            }
        }
    }
}
