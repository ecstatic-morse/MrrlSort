A stable, linearithmic sort in constant space written in Rust. Uses the method
described in ["Fast Stable Merging And Sorting In Constant Extra
Space"][grailsort] by Bing-chao Huang and Michael A. Langston. Heavily inspired
by (and named in honor of) Andrey Astrelin's [grailsort][mrrl].
Documented in more detail [in this blog post](https://ecstaticmorse.net/posts/grailsort/).

[grailsort]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.54.8381
[mrrl]: https://github.com/Mrrl/GrailSort

## TODO

This implementation needs some work before it's complete. Here are a few possible
improvements (most of them suggested by the folks behind the [HolyGrailSort
project](https://github.com/HolyGrailSortProject/Rewritten-Grailsort)).

Optimizations:
- [ ] Use MᴇʀɢᴇBᴜғRɪɢʜᴛ (working from right to left) for the last step in Phase 1.
- [ ] **IMPORTANT** During Phase 2, alternate between a mirrored version of the algorithm and
  the current version to avoid needing to rotate the internal buffer back to
  the start of the array at each step.
  - Another alternative (Kotasort?) is to *swap* the internal buffer into
    place, then fix up the swapped block's position during BʟᴏᴄᴋRᴏʟʟ.
- [ ] Faster sorting algorithm to prepare the movement imitation buffer:
  - Shell sort
  - Quick sort
  - Linear-time "deinterleave"
    - Relies on the fact that BʟᴏᴄᴋRᴏʟʟ does not change the relative order of A and B blocks; it merely interleves them.
    - This interleaving can be undone in linear time since we know the median element in the movement imitation buffer.
    - Not valid for [Phase
      2](https://ecstaticmorse.net/posts/grailsort/#phase-2-x-le-m--x2) with a
      the inline block-taggging scheme, since the movement imitation buffer is
      repurposed as the internal buffer for MᴇʀɢᴇBᴜғ. Would work for [Phase 3](https://ecstaticmorse.net/posts/grailsort/#phase-3-x2-le-m), though.
- [ ] Adaptive merging.
- [ ] Faster MᴇʀɢᴇBᴜғ implementation (especially for Phase 2).
  - Current version is a "tape" merge.
  - Could also use a "simple binary" merge: Do an exponential search at each step and swap in batches.
  - [Hwang-Lin](https://www.u-cursos.cl/ingenieria/2017/2/CC7310/1/material_docente/bajar?id_material=1894891), a happy medium between the two.
- [ ] Make use small allocations (e.g., big enough for the auxiliary buffer but less than N/2).

Features:
- [ ] Visualize with [`ArrayV`][] (or something else that interoperates with Rust).
- [ ] Parametric benchmarking:
  - Element size
  - Comparison cost
  - Number of distinct elements

[`ArrayV`]: https://github.com/Gaming32/ArrayV
