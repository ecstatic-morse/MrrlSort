#![feature(core_intrinsics)]
#![feature(slice_ptr_get, slice_ptr_len)]
#![cfg_attr(not(test), no_std)]

#[cfg(not(test))]
extern crate core as std;

/// Reborrows the slice in a `Blocks<&mut [T]>`.
macro_rules! rbrw {
    ($e:expr) => {
        Blocks {
            seq: &mut *$e.seq,
            ..$e
        }
    };
}

#[cfg(test)]
#[macro_use]
mod test_utils;

mod block;
mod extract_buf;
mod merge_buf;
mod merge_no_buf;
mod quadratic_sort;
mod search;
mod sort;
mod split;

pub use sort::{sort, sort_by};
use split::Split;

type Either<T = ()> = either::Either<T, T>;
