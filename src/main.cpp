// Huffman codec: command line front end and benchmark.
//
//   huffman                          run the benchmark with default settings
//   huffman bench [options]          same, explicit
//   huffman encode <in> [out]        compress a file
//   huffman decode <in> [out]        decompress a file
//
// See `huffman --help` for the full option list.

#include "huffman/huffman.hpp"
#include "huffman/timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------- statistics

/// Summary of a set of throughput samples. The mean and the spread are computed
/// over the inter-quartile range only, which discards scheduler and cache
/// outliers; `kept` reports how many samples survived that filter.
struct summary {
    double min{0};
    double q1{0};
    double median{0};
    double q3{0};
    double max{0};
    double mean{0};
    double stddev{0};
    std::size_t samples{0};
    std::size_t kept{0};

    [[nodiscard]] double iqr() const noexcept { return q3 - q1; }
};

[[nodiscard]] summary summarise(std::vector<double> samples)
{
    summary result;
    result.samples = samples.size();
    if (samples.empty()) {
        return result;
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    result.min = samples.front();
    result.max = samples.back();
    result.median = samples[n / 2];
    result.q1 = samples[n / 4];
    result.q3 = samples[(3 * n) / 4];

    // Second pass over the inter-quartile range.
    double sum = 0;
    double sum_sq = 0;
    for (const double value : samples) {
        if (value < result.q1 || value > result.q3) {
            continue;
        }
        sum += value;
        sum_sq += value * value;
        ++result.kept;
    }
    if (result.kept == 0) {
        result.kept = n;
        sum = 0;
        sum_sq = 0;
        for (const double value : samples) {
            sum += value;
            sum_sq += value * value;
        }
    }

    result.mean = sum / static_cast<double>(result.kept);
    const double variance = sum_sq / static_cast<double>(result.kept) - result.mean * result.mean;
    result.stddev = variance > 0 ? std::sqrt(variance) : 0;
    return result;
}

// ---------------------------------------------------------------- benchmark

struct bench_options {
    std::uint64_t bytes{1u << 20}; ///< 1 MiB
    std::uint32_t iterations{64};
    double p{0};          ///< fixed parameter; 0 means "draw one per iteration"
    double p_lo{0.25};    ///< lower bound of the drawn parameter
    double p_hi{0.75};    ///< upper bound of the drawn parameter
    std::uint32_t seed{12345};
    bool p_uniform{false}; ///< draw p from U(0, 1) instead of [p_lo, p_hi]
};

[[nodiscard]] std::uint64_t fnv1a(std::span<const std::uint8_t> data) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const std::uint8_t byte : data) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

int run_benchmark(const bench_options& options)
{
    std::vector<std::uint8_t> input(static_cast<std::size_t>(options.bytes));
    std::vector<std::uint8_t> encoded;
    encoded.reserve(input.size() / 4u + 64u);

    std::mt19937 engine{options.seed};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    std::vector<double> throughput;
    throughput.reserve(options.iterations);

    std::uint64_t checksum = 0;
    huffman::stopwatch watch;

    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        const double p = options.p_uniform
                             ? uniform(engine)
                             : (options.p > 0 ? options.p
                                              : options.p_lo + (options.p_hi - options.p_lo) * uniform(engine));
        std::geometric_distribution<int> distribution{std::max(p, std::numeric_limits<double>::min())};
        for (std::uint8_t& byte : input) {
            // The distribution is instantiated on int and clamped. Instantiating
            // it on std::uint8_t would be ill-formed (the standard requires an
            // integer type no smaller than short), and its behaviour above 255 is
            // implementation defined; clamping keeps the source well defined and
            // identical across standard libraries.
            byte = static_cast<std::uint8_t>(std::min(distribution(engine), 255));
        }
        checksum = fnv1a(input);

        watch.restart();
        huffman::encode_buffer(input, encoded);
        throughput.push_back(1e3 * static_cast<double>(input.size()) / watch.elapsed_ns());
    }

    const std::span<const std::uint8_t> input_span{input};
    const summary stats = summarise(throughput);

    // "best of" is the most stable estimator on a machine that is not quiet:
    // the fastest repetition is the one that suffered the least interference.
    // It repeats the *last* input only, so it is comparable across runs (and is
    // the number to quote when comparing two encoders), unlike the median above,
    // which averages over inputs of differing compressibility.
    constexpr int best_repetitions = 25;
    double best_ns = 1e300;
    for (int i = 0; i < best_repetitions; ++i) {
        watch.restart();
        huffman::encode_buffer(input, encoded);
        best_ns = std::min(best_ns, watch.elapsed_ns());
    }
    const double best_throughput = 1e3 * static_cast<double>(input.size()) / best_ns;

    std::cout << "Input:       " << input.size() << " bytes"
              << ", " << options.iterations << " iterations"
              << ", seed " << options.seed << '\n';
    std::cout << "Source:      ";
    if (options.p_uniform) {
        std::cout << "geometric, p ~ U(0, 1)";
    } else if (options.p > 0) {
        std::cout << "geometric, p = " << options.p;
    } else {
        std::cout << "geometric, p ~ U(" << options.p_lo << ", " << options.p_hi << ')';
    }
    std::cout << '\n';
    std::cout << "Data:        fnv1a = 0x" << std::hex << checksum << std::dec
              << ", alphabet = " << huffman::build_code_table(huffman::count_frequencies(input_span)).symbols
              << " symbols\n";
    std::cout << "Compressed:  " << encoded.size() << " bytes ("
              << (input.empty() ? 0.0 : 100.0 * static_cast<double>(encoded.size()) / static_cast<double>(input.size()))
              << " %)\n";

    std::cout << "Performance (histogram + model + payload, one input per iteration):\n";
    std::cout << "  median " << stats.median << " MB/s   IQR " << stats.iqr() << '\n';
    std::cout << "  mean   " << stats.mean << " MB/s   stddev " << stats.stddev << '\n';
    std::cout << "  min " << stats.min << "   max " << stats.max << "   (kept " << stats.kept << " of "
              << stats.samples << " samples)\n";
    std::cout << "  best   " << best_throughput << " MB/s   (" << best_ns * 1e-6 << " ms, best of "
              << best_repetitions << " on the last input)\n";

    // Phase breakdown on the last (fixed) input, minimum of a few repetitions.
    constexpr int repetitions = 5;
    double histogram_ns = 1e300;
    double model_ns = 1e300;
    double payload_ns = 1e300;
    for (int i = 0; i < repetitions; ++i) {
        watch.restart();
        const huffman::histogram freq = huffman::count_frequencies(input_span);
        histogram_ns = std::min(histogram_ns, watch.elapsed_ns());

        watch.restart();
        const huffman::code_table table = huffman::build_code_table(freq);
        model_ns = std::min(model_ns, watch.elapsed_ns());

        watch.restart();
        encoded.clear();
        huffman::encode_payload(table, input_span, huffman::encoded_bits(table, freq), encoded);
        payload_ns = std::min(payload_ns, watch.elapsed_ns());
    }

    const double bytes = static_cast<double>(input.size());
    const auto report = [&](const char* name, double ns, bool rate) {
        std::cout << "  " << name << "  " << ns * 1e-6 << " ms";
        if (rate && ns > 0) {
            std::cout << "   " << 1e3 * bytes / ns << " MB/s";
        }
        std::cout << '\n';
    };
    std::cout << "Phase breakdown (best of " << repetitions << "):\n";
    report("histogram", histogram_ns, true);
    report("model    ", model_ns, false);
    report("payload  ", payload_ns, true);

    return 0;
}

// ---------------------------------------------------------------- file tools

[[nodiscard]] std::vector<std::uint8_t> read_file(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("cannot open '" + path.string() + "' for reading");
    }
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

void write_file(const std::filesystem::path& path, std::span<const std::uint8_t> data)
{
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw std::runtime_error("cannot open '" + path.string() + "' for writing");
    }
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) {
        throw std::runtime_error("failed to write '" + path.string() + "'");
    }
}

int run_encode(const std::filesystem::path& source, const std::filesystem::path& target, bool verify)
{
    const std::vector<std::uint8_t> input = read_file(source);
    const std::vector<std::uint8_t> encoded = huffman::compress(input);
    write_file(target, encoded);

    const double ratio = input.empty() ? 0.0
                                       : 100.0 * static_cast<double>(encoded.size()) / static_cast<double>(input.size());
    std::cout << source.string() << " -> " << target.string() << ": " << input.size() << " -> "
              << encoded.size() << " bytes (" << ratio << " %)\n";

    if (verify) {
        const std::vector<std::uint8_t> restored = huffman::decompress(encoded);
        if (restored != input) {
            throw std::runtime_error("round trip mismatch");
        }
        std::cout << "round trip verified\n";
    }
    return 0;
}

int run_decode(const std::filesystem::path& source, const std::filesystem::path& target)
{
    const std::vector<std::uint8_t> input = read_file(source);
    const std::vector<std::uint8_t> restored = huffman::decompress(input);
    write_file(target, restored);
    std::cout << source.string() << " -> " << target.string() << ": " << input.size() << " -> "
              << restored.size() << " bytes\n";
    return 0;
}

// ---------------------------------------------------------------- command line

void print_usage(std::ostream& out)
{
    out << "huffman - canonical Huffman codec and benchmark\n"
           "\n"
           "usage:\n"
           "  huffman [bench] [options]      run the throughput benchmark (default)\n"
           "  huffman encode <in> [out] [-v] compress a file\n"
           "  huffman decode <in> [out]      decompress a file\n"
           "  huffman --help\n"
           "\n"
           "bench options:\n"
           "  --bytes N        input size in bytes (default 1048576)\n"
           "  --mb N           input size in MiB (overrides --bytes)\n"
           "  --iters N        measured iterations (default 64)\n"
           "  --p P            fixed geometric parameter, e.g. 0.3\n"
           "  --p-range A B    draw p from [A, B] per iteration (default 0.25 0.75)\n"
           "  --seed N         random seed (default 12345)\n"
           "  --p-uniform      draw p from U(0, 1) per iteration\n"
           "\n"
           "encode options:\n"
           "  -v, --verify     decode the result and compare it with the input\n";
}

[[nodiscard]] std::string_view need_value(int argc, char** argv, int& i)
{
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string{"missing value for "} + argv[i]);
    }
    return argv[++i];
}

int run_command(int argc, char** argv)
{
    std::vector<std::string> args{argv, argv + argc};
    if (args.size() < 2) {
        return run_benchmark(bench_options{});
    }

    const std::string command = args[1];
    if (command == "--help" || command == "-h" || command == "help") {
        print_usage(std::cout);
        return 0;
    }

    if (command == "encode" || command == "decode") {
        const bool encode = command == "encode";
        std::string source;
        std::string target;
        bool verify = false;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const std::string& arg = args[i];
            if (arg == "-v" || arg == "--verify") {
                verify = true;
            } else if (source.empty()) {
                source = arg;
            } else if (target.empty()) {
                target = arg;
            } else {
                throw std::runtime_error("unexpected argument '" + arg + "'");
            }
        }
        if (source.empty()) {
            throw std::runtime_error("missing input file");
        }
        if (target.empty()) {
            target = encode ? source + ".huf" : std::filesystem::path{source}.replace_extension().string();
            if (!encode && target == source) {
                target = source + ".out";
            }
        }
        return encode ? run_encode(source, target, verify) : run_decode(source, target);
    }

    // Anything else is treated as benchmark options, so `huffman --iters 10` works.
    bench_options options;
    int index = 1;
    for (; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "bench") {
            continue;
        }
        if (arg == "--bytes") {
            options.bytes = std::stoull(std::string{need_value(argc, argv, index)});
        } else if (arg == "--mb") {
            options.bytes = std::stoull(std::string{need_value(argc, argv, index)}) << 20;
        } else if (arg == "--iters") {
            options.iterations = static_cast<std::uint32_t>(std::stoul(std::string{need_value(argc, argv, index)}));
        } else if (arg == "--p") {
            options.p = std::stod(std::string{need_value(argc, argv, index)});
        } else if (arg == "--p-range") {
            options.p_lo = std::stod(std::string{need_value(argc, argv, index)});
            options.p_hi = std::stod(std::string{need_value(argc, argv, index)});
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(std::string{need_value(argc, argv, index)}));
        } else if (arg == "--p-uniform") {
            options.p_uniform = true;
        } else {
            throw std::runtime_error("unknown option '" + std::string{arg} + "'");
        }
    }

    if (options.iterations == 0) {
        throw std::runtime_error("--iters must be at least 1");
    }
    if (options.p < 0 || options.p >= 1 || options.p_lo <= 0 || options.p_hi >= 1 || options.p_lo > options.p_hi) {
        throw std::runtime_error("the geometric parameter must be in (0, 1)");
    }
    return run_benchmark(options);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        return run_command(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
