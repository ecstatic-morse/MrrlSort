use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

fn random_vec<T>(len: usize, seed: u64) -> Vec<T>
where
    rand::distributions::Standard: Distribution<T>,
{
    let rng = SmallRng::seed_from_u64(seed);
    rng.sample_iter(rand::distributions::Standard)
        .take(len)
        .collect()
}

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_sort");

    // Log-scale
    let plot_config =
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let sizes = (10..=22).step_by(3).map(|x| 1 << x);
    for size in sizes {
        let batch_size = if size > (1 << 20) {
            BatchSize::LargeInput
        } else {
            BatchSize::SmallInput
        };

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &size| {
            let input = random_vec::<u8>(size, 42);
            b.iter_batched(|| input.clone(), |mut v| v.sort(), batch_size);
        });

        group.bench_with_input(BenchmarkId::new("block_sort", size), &size, |b, &size| {
            let input = random_vec::<u8>(size, 42);
            b.iter_batched(
                || input.clone(),
                |mut v| block_sort::sort(&mut v),
                batch_size,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
