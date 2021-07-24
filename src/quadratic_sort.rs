//! In-place sorting algorithms that take quadratic time.

use super::search::binary_search;

pub fn insertion_sort<T>(v: &mut [T], is_less: &mut impl FnMut(&T, &T) -> bool) {
    for i in 1..v.len() {
        std_impl::shift_tail(&mut v[..i + 1], is_less);
    }
}

#[allow(unused)]
pub fn insertion_sort_binary<T>(v: &mut [T], is_less: &mut impl FnMut(&T, &T) -> bool) {
    for i in 1..v.len() {
        let sorted = &v[..i];
        let el = &v[i];

        let pos = binary_search(sorted, |x| is_less(x, el));
        if pos != i {
            v[pos..=i].rotate_right(1);
        }
    }
}

/// Functionality taken directly from the standard library.
///
/// Everything here has been copy-pasted exactly, so it should be sound.
mod std_impl {
    use std::{mem, ptr};

    /// When dropped, copies from `src` into `dest`.
    struct CopyOnDrop<T> {
        src: *mut T,
        dest: *mut T,
    }

    impl<T> Drop for CopyOnDrop<T> {
        fn drop(&mut self) {
            // SAFETY:  This is a helper class.
            //          Please refer to its usage for correctness.
            //          Namely, one must be sure that `src` and `dst` does not overlap as required by `ptr::copy_nonoverlapping`.
            unsafe {
                ptr::copy_nonoverlapping(self.src, self.dest, 1);
            }
        }
    }

    /// Shifts the last element to the left until it encounters a smaller or equal element.
    pub fn shift_tail<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = v.len();
        // SAFETY: The unsafe operations below involves indexing without a bound check (`get_unchecked` and `get_unchecked_mut`)
        // and copying memory (`ptr::copy_nonoverlapping`).
        //
        // a. Indexing:
        //  1. We checked the size of the array to >= 2.
        //  2. All the indexing that we will do is always between `0 <= index < len-1` at most.
        //
        // b. Memory copying
        //  1. We are obtaining pointers to references which are guaranteed to be valid.
        //  2. They cannot overlap because we obtain pointers to difference indices of the slice.
        //     Namely, `i` and `i+1`.
        //  3. If the slice is properly aligned, the elements are properly aligned.
        //     It is the caller's responsibility to make sure the slice is properly aligned.
        //
        // See comments below for further detail.
        unsafe {
            // If the last two elements are out-of-order...
            if len >= 2 && is_less(v.get_unchecked(len - 1), v.get_unchecked(len - 2)) {
                // Read the last element into a stack-allocated variable. If a following comparison
                // operation panics, `hole` will get dropped and automatically write the element back
                // into the slice.
                let mut tmp = mem::ManuallyDrop::new(ptr::read(v.get_unchecked(len - 1)));
                let mut hole = CopyOnDrop {
                    src: &mut *tmp,
                    dest: v.get_unchecked_mut(len - 2),
                };
                ptr::copy_nonoverlapping(v.get_unchecked(len - 2), v.get_unchecked_mut(len - 1), 1);

                for i in (0..len - 2).rev() {
                    if !is_less(&*tmp, v.get_unchecked(i)) {
                        break;
                    }

                    // Move `i`-th element one place to the right, thus shifting the hole to the left.
                    ptr::copy_nonoverlapping(v.get_unchecked(i), v.get_unchecked_mut(i + 1), 1);
                    hole.dest = v.get_unchecked_mut(i);
                }
                // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::is_sorted;
    use proptest::prelude::*;

    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort_unstable();
        v
    }

    proptest! {
        #[test]
        fn insertion_sort_binary(mut arr in proptest::collection::vec(0..30, 0..100).prop_map(sorted)) {
            super::insertion_sort_binary(&mut arr, &mut std::cmp::PartialOrd::lt);
            prop_assert!(is_sorted(&arr));
        }
    }
}
