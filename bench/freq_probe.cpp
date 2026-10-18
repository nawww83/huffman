// Effective-frequency probe for the benchmark.
//
// Measures a long serial chain of dependent operations and prints how many
// iterations the CPU retired per nanosecond. On a turbo-capable part that ratio
// is proportional to the current clock, so it can be used to normalise
// throughput measured at different frequencies:
//
//     MB/s at reference clock = measured MB/s * (probe_now / probe_reference)
//
// Run it pinned to the same core as the benchmark, immediately before and after
// each measurement, and average the two probes. See the benchmarking section of
// README.md and `measure.ps1`.
//
// The absolute number is only meaningful as a *relative* frequency indicator
// (the cycle cost of one iteration is a CPU-specific constant), which is why the
// program prints the ratio rather than a frequency in GHz.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// Keeps the chain dependent and stops the optimiser from folding it away. The
// volatile fallback costs a few cycles per iteration but stays proportional to
// the clock, which is all the probe needs.
inline void keep_alive(std::uint64_t& value) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : "+r"(value));
#else
    volatile std::uint64_t sink = value;
    value = sink;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    const std::uint64_t iterations = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200'000'000ull;

    std::uint64_t value = 1;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        value = value + 1;
        keep_alive(value);
    }
    const auto stop = std::chrono::steady_clock::now();

    const double nanoseconds =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    const double ops_per_ns = static_cast<double>(iterations) / nanoseconds;

    std::printf("ops_per_ns %.4f\n", ops_per_ns);
    return value == 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
