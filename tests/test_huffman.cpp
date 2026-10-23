// Test suite for the Huffman codec. Exits with a non-zero status on failure,
// so CTest picks the result up directly.
//
// Huffman coding is sensitive to edge cases, so the sweeps below cover:
//   * alphabet sizes 0, 1, 2, 3, up to 256;
//   * flat, geometric, "one dominant symbol", power-of-two and Fibonacci-like
//     frequencies (the last one drives the tree far past the code length limit);
//   * inputs of size 0..8 (the bit padding boundaries) and larger blocks;
//   * random histograms, as a property-based check.

#include "huffman/bit_stream.hpp"
#include "huffman/huffman.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void report(bool ok, const char* expression, const char* file, int line)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %s:%d: %s\n", file, line, expression);
    }
}

template <typename A, typename B>
void report_equal(const A& lhs, const B& rhs, const char* expression, const char* file, int line)
{
    report(lhs == rhs, expression, file, line);
}

#define CHECK(expression) report((expression), #expression, __FILE__, __LINE__)
#define CHECK_EQ(lhs, rhs) report_equal((lhs), (rhs), #lhs " == " #rhs, __FILE__, __LINE__)

// ---------------------------------------------------------- table invariants

/// No code word may be a prefix of another one.
[[nodiscard]] bool is_prefix_free(const huffman::code_table& table)
{
    for (std::size_t outer = 0; outer < huffman::alphabet_size; ++outer) {
        if (table.len[outer] == 0u) {
            continue;
        }
        for (std::size_t inner = 0; inner < huffman::alphabet_size; ++inner) {
            if (inner == outer || table.len[inner] <= table.len[outer]) {
                continue;
            }
            const auto shift = table.len[inner] - table.len[outer];
            if ((table.code[inner] >> shift) == table.code[outer]) {
                return false;
            }
        }
    }
    return true;
}

/// A complete prefix code satisfies the Kraft equality with sum 1.
[[nodiscard]] bool kraft_is_saturated(const huffman::code_table& table)
{
    std::uint64_t sum = 0;
    for (const std::uint8_t len : table.len) {
        if (len != 0u) {
            sum += std::uint64_t{1} << (table.max_len - len);
        }
    }
    return sum == (std::uint64_t{1} << table.max_len);
}

/// Every code must fit into its own length.
[[nodiscard]] bool codes_fit_their_length(const huffman::code_table& table)
{
    for (std::size_t byte = 0; byte < huffman::alphabet_size; ++byte) {
        const unsigned len = table.len[byte];
        if (len != 0u && len < 32u && (table.code[byte] >> len) != 0u) {
            return false;
        }
    }
    return true;
}

/// `symbols` must match the number of non-zero lengths and `max_len` the longest.
[[nodiscard]] bool bookkeeping_is_consistent(const huffman::code_table& table)
{
    std::size_t present = 0;
    unsigned longest = 0;
    for (const std::uint8_t len : table.len) {
        if (len != 0u) {
            ++present;
            longest = std::max(longest, static_cast<unsigned>(len));
        }
    }
    return present == table.symbols && longest == table.max_len;
}

void check_table(const huffman::code_table& table, const char* what, const char* file, int line)
{
    const bool ok = codes_fit_their_length(table) &&
                    bookkeeping_is_consistent(table) &&
                    table.max_len <= huffman::max_code_length &&
                    (table.symbols < 2u || (is_prefix_free(table) && kraft_is_saturated(table)));
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %s:%d: invalid code table [%s] (symbols=%u max_len=%u)\n", file, line, what,
                    static_cast<unsigned>(table.symbols), static_cast<unsigned>(table.max_len));
    }
}

#define CHECK_TABLE(table, what) check_table((table), (what), __FILE__, __LINE__)

/// Straightforward bit-by-bit canonical decoder. It shares no code with the
/// table-driven decoder under test, so it works as an oracle for it: in
/// particular for the packed multi-symbol entries and for the tail.
[[nodiscard]] std::vector<std::uint8_t> reference_decode(const huffman::code_table& table,
                                                        std::span<const std::uint8_t> payload,
                                                        std::size_t symbol_count)
{
    std::array<std::uint8_t, huffman::alphabet_size> order{};
    std::array<std::uint32_t, huffman::alphabet_size + 1> count{};
    std::size_t next = 0;
    for (unsigned len = 1; len <= table.max_len; ++len) {
        for (std::size_t byte = 0; byte < huffman::alphabet_size; ++byte) {
            if (table.len[byte] == len) {
                order[next++] = static_cast<std::uint8_t>(byte);
                ++count[len];
            }
        }
    }

    const std::size_t total_bits = payload.size() * 8u;
    std::vector<std::uint8_t> out;
    out.reserve(symbol_count);

    std::size_t bit = 0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        std::uint32_t code = 0;
        std::uint32_t first = 0;
        std::size_t index = 0;
        bool found = false;
        for (unsigned len = 1; len <= table.max_len; ++len) {
            const bool one = bit < total_bits && ((payload[bit >> 3] >> (7u - (bit & 7u))) & 1u) != 0u;
            ++bit;
            code = (code << 1u) | (one ? 1u : 0u);
            if (code - first < count[len]) {
                out.push_back(order[index + (code - first)]);
                found = true;
                break;
            }
            index += count[len];
            first = (first + count[len]) << 1u;
        }
        if (!found) {
            break; // the bit stream is not a code word; the caller notices the short result
        }
    }
    return out;
}

// ------------------------------------------------------------- round tripping

[[nodiscard]] bool same_bytes(std::span<const std::uint8_t> lhs, std::span<const std::uint8_t> rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

/// Checks the table, the payload, the padding and both decode paths.
void check_round_trip(std::span<const std::uint8_t> input, const char* what, const char* file, int line)
{
    const auto expect = [&](bool ok, const char* detail) {
        ++g_checks;
        if (!ok) {
            ++g_failures;
            std::printf("  FAIL %s:%d: %s [%s]\n", file, line, detail, what);
        }
    };

    const huffman::histogram freq = huffman::count_frequencies(input);
    const huffman::code_table table = huffman::build_code_table(freq);
    check_table(table, what, file, line);

    const std::size_t bits = huffman::encoded_bits(table, input);
    expect(bits == huffman::encoded_bits(table, freq), "bit count differs between input and histogram");

    std::vector<std::uint8_t> payload;
    huffman::encode_payload(table, input, bits, payload);
    expect(payload.size() == huffman::payload_size(bits), "payload size does not match the bit count");

    if (!payload.empty() && bits % 8u != 0u) {
        const auto pad = static_cast<unsigned>(8u - bits % 8u);
        expect((payload.back() & huffman::low_mask(pad)) == 0u, "padding bits are not zero");
    }

    const std::vector<std::uint8_t> encoded = huffman::compress(input);
    expect(encoded.size() == huffman::container_header_size(table.symbols) + payload.size(),
           "container size is not header + payload");
    expect(same_bytes(huffman::decompress(encoded), input), "container round trip");

    std::vector<std::uint8_t> restored;
    huffman::decode_payload(table, payload, input.size(), restored);
    expect(same_bytes(restored, input), "payload round trip");

    // The convenience overload must agree with the explicit bit count.
    std::vector<std::uint8_t> convenience;
    huffman::encode_payload(table, input, convenience);
    expect(convenience == payload, "encode_payload overloads disagree");

    // The container must be deterministic.
    expect(huffman::compress(input) == encoded, "compress is not deterministic");
}

#define CHECK_ROUND_TRIP(input, what) check_round_trip((input), (what), __FILE__, __LINE__)

// ------------------------------------------------------------------- sources

enum class shape {
    empty,
    constant,
    two_values,
    uniform_full,
    uniform_narrow,
    geometric,
    dominant,
    power_law,
    alternating,
};

/// Builds an input with the requested frequency shape.
[[nodiscard]] std::vector<std::uint8_t> make_input(shape kind, std::size_t size, std::mt19937& engine)
{
    std::vector<std::uint8_t> input(size);
    if (kind == shape::empty) {
        input.clear();
        return input;
    }
    if (kind == shape::constant) {
        std::fill(input.begin(), input.end(), std::uint8_t{0xA5});
        return input;
    }
    if (kind == shape::two_values) {
        for (std::size_t i = 0; i < size; ++i) {
            input[i] = (engine() % 4u == 0u) ? std::uint8_t{0x11} : std::uint8_t{0x22};
        }
        return input;
    }
    if (kind == shape::uniform_full) {
        std::uniform_int_distribution<int> distribution{0, 255};
        for (std::uint8_t& byte : input) {
            byte = static_cast<std::uint8_t>(distribution(engine));
        }
        return input;
    }
    if (kind == shape::uniform_narrow) {
        const auto width = static_cast<int>(2u + engine() % 20u);
        std::uniform_int_distribution<int> distribution{0, width - 1};
        for (std::uint8_t& byte : input) {
            byte = static_cast<std::uint8_t>(distribution(engine));
        }
        return input;
    }
    if (kind == shape::geometric) {
        const double p = 0.1 + 0.8 * (static_cast<double>(engine() % 1000u) / 1000.0);
        std::geometric_distribution<int> distribution{p};
        for (std::uint8_t& byte : input) {
            byte = static_cast<std::uint8_t>(std::min(distribution(engine), 255));
        }
        return input;
    }
    if (kind == shape::dominant) {
        std::uniform_int_distribution<int> rare{0, 255};
        for (std::uint8_t& byte : input) {
            byte = (engine() % 10u == 0u) ? static_cast<std::uint8_t>(rare(engine)) : std::uint8_t{7};
        }
        return input;
    }
    if (kind == shape::power_law) {
        // Symbol 0 dominates and the tail decays by 2^3 per symbol, so the
        // tree has one very short code next to a long, thin staircase.
        std::array<std::uint64_t, 256> weight{};
        for (std::size_t i = 0; i < 21; ++i) {
            weight[i] = std::uint64_t{1} << (60u - 3u * i);
        }
        std::uint64_t total = 0;
        for (const std::uint64_t value : weight) {
            total += value;
        }
        std::uniform_int_distribution<std::uint64_t> pick{0, total - 1};
        for (std::uint8_t& byte : input) {
            const std::uint64_t value = pick(engine);
            std::uint64_t acc = 0;
            std::size_t symbol = 0;
            for (; symbol < 255u; ++symbol) {
                if (value < acc + weight[symbol]) {
                    break;
                }
                acc += weight[symbol];
            }
            byte = static_cast<std::uint8_t>(symbol);
        }
        return input;
    }
    if (kind == shape::alternating) {
        for (std::size_t i = 0; i < size; ++i) {
            input[i] = static_cast<std::uint8_t>(i % 251u);
        }
        return input;
    }
    return input;
}

constexpr std::array<shape, 9> all_shapes{
    shape::empty,     shape::constant,    shape::two_values, shape::uniform_full, shape::uniform_narrow,
    shape::geometric, shape::dominant,    shape::power_law,  shape::alternating,
};

[[nodiscard]] const char* name_of(shape kind)
{
    switch (kind) {
    case shape::empty: return "empty";
    case shape::constant: return "constant";
    case shape::two_values: return "two_values";
    case shape::uniform_full: return "uniform_full";
    case shape::uniform_narrow: return "uniform_narrow";
    case shape::geometric: return "geometric";
    case shape::dominant: return "dominant";
    case shape::power_law: return "power_law";
    case shape::alternating: return "alternating";
    }
    return "?";
}

// ------------------------------------------------------------------- tests

void test_bit_stream_round_trip()
{
    std::mt19937 engine{20240607};
    std::vector<std::uint8_t> buffer;

    for (int trial = 0; trial < 400; ++trial) {
        const int count = 1 + static_cast<int>(engine() % 64u);
        std::vector<std::pair<std::uint32_t, unsigned>> words;
        std::size_t total_bits = 0;
        for (int i = 0; i < count; ++i) {
            const unsigned len = 1u + (engine() % 32u);
            const std::uint32_t value = engine() & huffman::low_mask(len);
            words.emplace_back(value, len);
            total_bits += len;
        }

        buffer.assign(huffman::payload_size(total_bits), 0);
        {
            huffman::bit_writer writer{buffer.data()};
            for (const auto& [value, len] : words) {
                writer.put(value, len);
            }
            writer.flush();
            CHECK_EQ(writer.bytes_written(), buffer.size());
        }

        huffman::bit_reader reader{buffer};
        for (const auto& [value, len] : words) {
            CHECK_EQ(reader.get(len), value);
        }
        buffer.clear();
    }
}

void test_canonical_codes()
{
    // frequencies: 'a' = 3, 'b' = 2, 'c' = 1  ->  lengths 1, 2, 2
    huffman::histogram freq{};
    freq['a'] = 3;
    freq['b'] = 2;
    freq['c'] = 1;
    const huffman::code_table table = huffman::build_code_table(freq);

    CHECK_EQ(table.symbols, 3);
    CHECK_EQ(table.max_len, 2);
    CHECK_EQ(table.len['a'], 1);
    CHECK_EQ(table.len['b'], 2);
    CHECK_EQ(table.len['c'], 2);
    CHECK_EQ(table.code['a'], 0u);
    CHECK_EQ(table.code['b'], 2u);
    CHECK_EQ(table.code['c'], 3u);
    CHECK_TABLE(table, "aaabbc");

    // "aaabbc" -> 0 0 0 10 10 11 = 000101011 -> 0x15, 0x80
    const std::string text = "aaabbc";
    std::vector<std::uint8_t> payload;
    huffman::encode_payload(table,
                            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()),
                                                          text.size()),
                            payload);
    CHECK_EQ(payload, (std::vector<std::uint8_t>{0x15, 0x80}));

    std::vector<std::uint8_t> restored;
    huffman::decode_payload(table, payload, text.size(), restored);
    CHECK_EQ(std::string(restored.begin(), restored.end()), text);
}

void test_code_table_invariants()
{
    std::mt19937 engine{7};

    // Flat histograms of every alphabet size: the widest spread of collisions.
    for (std::size_t symbols = 1; symbols <= huffman::alphabet_size; ++symbols) {
        huffman::histogram freq{};
        for (std::size_t i = 0; i < symbols; ++i) {
            freq[i] = 1;
        }
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.symbols, symbols);
        CHECK_TABLE(table, "flat histogram");
        if (symbols > 1u) {
            // A flat histogram is always a balanced, complete code.
            const unsigned expected = static_cast<unsigned>(std::bit_width(symbols - 1u));
            CHECK(table.max_len == expected || table.max_len == expected + 1u);
        }
    }

    // Two symbols with equal counts must get one bit each.
    {
        huffman::histogram freq{};
        freq[10] = 5;
        freq[11] = 5;
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.max_len, 1);
        CHECK_EQ(table.symbols, 2);
        CHECK_TABLE(table, "two symbols");
    }

    // A single dominant symbol surrounded by a long tail.
    {
        huffman::histogram freq{};
        freq[0] = std::uint64_t{1} << 40;
        for (std::size_t i = 1; i < 256; ++i) {
            freq[i] = i;
        }
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.symbols, 256);
        CHECK_EQ(table.len[0], 1);
        CHECK_TABLE(table, "dominant symbol");
    }

    // Frequencies spanning 62 binary orders of magnitude: the tree would be far
    // deeper than the limit if the builder did not flatten it.
    {
        huffman::histogram freq{};
        for (std::size_t i = 0; i < 40; ++i) {
            freq[i] = std::uint64_t{1} << (i * 62u / 39u);
        }
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.symbols, 40);
        CHECK(table.max_len <= huffman::max_code_length);
        CHECK_TABLE(table, "wide dynamic range");
    }

    // Extremes of the counter type.
    {
        huffman::histogram freq{};
        freq[0] = std::numeric_limits<std::uint64_t>::max() / 2u;
        freq[1] = std::numeric_limits<std::uint64_t>::max() / 2u;
        freq[2] = 1;
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.symbols, 3);
        CHECK_TABLE(table, "counter extremes");
    }
}

void test_length_limited_codes()
{
    // Fibonacci-like counts give the most unbalanced tree possible, whose depth
    // grows with the number of symbols. The builder must flatten the histogram
    // until the limit is met instead of emitting codes it cannot represent.
    huffman::histogram freq{};
    std::uint64_t previous = 1;
    std::uint64_t current = 1;
    for (std::size_t symbol = 0; symbol < 60; ++symbol) {
        freq[symbol] = previous;
        const std::uint64_t next = previous + current;
        previous = current;
        current = next;
    }

    const huffman::code_table table = huffman::build_code_table(freq);
    CHECK_EQ(table.symbols, 60);
    CHECK(table.max_len <= huffman::max_code_length);
    CHECK_TABLE(table, "fibonacci counts");

    // The same shape, but with counts that fit an actual input: the rare symbols
    // must still round trip.
    std::vector<std::uint8_t> input;
    for (std::size_t symbol = 0; symbol < 60; ++symbol) {
        input.insert(input.end(), 60u - symbol, static_cast<std::uint8_t>(symbol));
    }
    const std::vector<std::uint8_t> base = input;
    for (int repeat = 0; repeat < 8; ++repeat) {
        input.insert(input.end(), base.begin(), base.end());
    }
    CHECK_ROUND_TRIP(input, "skewed 60 symbol input");
}

void test_degenerate_alphabet()
{
    // Empty input.
    {
        const huffman::code_table table = huffman::build_code_table(huffman::histogram{});
        CHECK(table.degenerate());
        CHECK_EQ(table.symbols, 0);
        CHECK_EQ(table.max_len, 0);
        std::vector<std::uint8_t> payload;
        huffman::encode_payload(table, {}, payload);
        CHECK(payload.empty());
        std::vector<std::uint8_t> restored{1, 2, 3};
        huffman::decode_payload(table, payload, 0, restored);
        CHECK(restored.empty());
    }
    // Single symbol: no bits are emitted.
    {
        huffman::histogram freq{};
        freq['q'] = 12345;
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK(table.degenerate());
        CHECK_EQ(table.symbols, 1);
        CHECK_EQ(table.only_symbol(), static_cast<std::uint8_t>('q'));
        CHECK_TABLE(table, "single symbol");
        CHECK_EQ(huffman::encoded_bits(table, std::vector<std::uint8_t>{'q'}), 0u);

        const std::vector<std::uint8_t> input(1000, static_cast<std::uint8_t>('q'));
        std::vector<std::uint8_t> payload;
        huffman::encode_payload(table, input, payload);
        CHECK(payload.empty());

        std::vector<std::uint8_t> restored;
        huffman::decode_payload(table, payload, input.size(), restored);
        CHECK_EQ(restored, input);
    }
}

void test_sizes_and_padding()
{
    std::mt19937 engine{99};

    // Sizes around the byte and word boundaries: this is where the writer's
    // partial-word handling and the padding show up.
    for (const std::size_t size : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u,
                                   64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u, 1000u, 4096u}) {
        for (const shape kind : all_shapes) {
            const std::vector<std::uint8_t> input = make_input(kind, size, engine);
            CHECK_ROUND_TRIP(input, name_of(kind));
        }
    }

    // A single byte of every value, so that every code of the alphabet is used.
    {
        std::vector<std::uint8_t> input;
        for (unsigned value = 0; value < 256; ++value) {
            input.push_back(static_cast<std::uint8_t>(value));
        }
        CHECK_ROUND_TRIP(input, "one of each byte");
        CHECK_EQ(huffman::compress(input).size(),
                 huffman::container_header_size(256) + huffman::payload_size(8u * 256u));
    }
}

void test_decode_matches_reference()
{
    std::mt19937 engine{0xDEC0DE};

    // The decoder has three regimes that must agree with the oracle: entries
    // that carry several symbols, entries that carry one, and the tail (last
    // symbols and last payload bytes). Sizes 0..40 step through all of them.
    std::vector<std::size_t> sizes;
    for (std::size_t size = 0; size <= 40; ++size) {
        sizes.push_back(size);
    }
    for (const std::size_t size : {63u, 64u, 65u, 127u, 128u, 255u, 1000u, 5000u}) {
        sizes.push_back(size);
    }

    for (const std::size_t size : sizes) {
        for (const shape kind : all_shapes) {
            const std::vector<std::uint8_t> input = make_input(kind, size, engine);
            const huffman::code_table table = huffman::build_code_table(huffman::count_frequencies(input));
            const std::size_t bits = huffman::encoded_bits(table, input);

            std::vector<std::uint8_t> payload;
            huffman::encode_payload(table, input, bits, payload);

            std::vector<std::uint8_t> decoded;
            huffman::decode_payload(table, payload, input.size(), decoded);

            const std::vector<std::uint8_t> oracle = reference_decode(table, payload, input.size());
            const bool ok = oracle == input && decoded == oracle;
            ++g_checks;
            if (!ok) {
                ++g_failures;
                std::printf("  FAIL %s:%d: decoder disagrees with the reference [%s, %zu bytes]\n", __FILE__,
                            __LINE__, name_of(kind), size);
            }
        }
    }

    // Codes longer than the root table: every such symbol goes through the
    // bit-by-bit walk in the middle of a packed stream. Fibonacci counts give
    // the deepest tree the length limit allows, and the table is then used to
    // encode an input that is small enough for a test.
    {
        huffman::histogram freq{};
        std::uint64_t previous = 1;
        std::uint64_t current = 1;
        for (std::size_t symbol = 0; symbol < 40; ++symbol) {
            freq[symbol] = previous;
            const std::uint64_t next = previous + current;
            previous = current;
            current = next;
        }
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK(table.max_len > 12u);
        CHECK_TABLE(table, "fibonacci table");

        std::vector<std::uint8_t> input;
        input.reserve(4096);
        for (std::size_t i = 0; i < 4000; ++i) {
            input.push_back(static_cast<std::uint8_t>(i % 40u));
        }
        std::vector<std::uint8_t> payload;
        huffman::encode_payload(table, input, payload);
        std::vector<std::uint8_t> decoded;
        huffman::decode_payload(table, payload, input.size(), decoded);
        CHECK(decoded == input);
        CHECK(reference_decode(table, payload, input.size()) == input);
    }

    // Every symbol of a full alphabet: 256 codes of eight bits, the maximum
    // number of distinct entries a root table can hold.
    {
        std::vector<std::uint8_t> input;
        for (unsigned value = 0; value < 256; ++value) {
            input.push_back(static_cast<std::uint8_t>(value));
        }
        const std::vector<std::uint8_t> block = input;
        for (unsigned repeat = 0; repeat < 64; ++repeat) {
            input.insert(input.end(), block.begin(), block.end());
        }
        const huffman::code_table table = huffman::build_code_table(huffman::count_frequencies(input));
        CHECK_EQ(table.max_len, 8u);
        std::vector<std::uint8_t> payload;
        huffman::encode_payload(table, input, payload);
        std::vector<std::uint8_t> decoded;
        huffman::decode_payload(table, payload, input.size(), decoded);
        CHECK(decoded == input);
        CHECK(reference_decode(table, payload, input.size()) == input);
    }
}

void test_decode_rejects_incomplete_code()
{
    const auto throws = [](const huffman::code_table& table, std::span<const std::uint8_t> payload,
                           std::size_t symbols) {
        try {
            std::vector<std::uint8_t> out;
            huffman::decode_payload(table, payload, symbols, out);
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };

    // An incomplete (Kraft sum < 1) code: 26 symbols of five bits leave most
    // bit patterns without a code word, so 0xFF cannot be decoded.
    huffman::histogram freq{};
    for (std::size_t symbol = 0; symbol < 26; ++symbol) {
        freq['a' + symbol] = 1;
    }
    huffman::code_table table = huffman::build_code_table(freq);
    table.len.fill(0); // rebuild by hand: a deliberately incomplete code
    for (std::size_t symbol = 0; symbol < 26; ++symbol) {
        table.len['a' + symbol] = 5;
    }
    table.symbols = 26;
    huffman::assign_canonical_codes(table);
    CHECK_EQ(table.max_len, 5u);

    const std::vector<std::uint8_t> garbage(64, 0xFF);
    // The tail path and the packed-entry path must both give up instead of
    // reading past the end of the payload.
    CHECK(throws(table, garbage, 3));
    CHECK(throws(table, garbage, 300));
    CHECK(throws(table, garbage, 1));

    // A bit pattern that is a code word still decodes.
    const std::vector<std::uint8_t> valid(64, 0x00); // 00000 -> the first symbol
    std::vector<std::uint8_t> out;
    huffman::decode_payload(table, valid, 100, out);
    CHECK_EQ(out.size(), 100u);
}

void test_random_fuzz()
{
    std::mt19937 engine{0xC0FFEE};

    for (int trial = 0; trial < 400; ++trial) {
        const shape kind = all_shapes[engine() % all_shapes.size()];
        const auto size = static_cast<std::size_t>(1u + engine() % 20000u);
        const std::vector<std::uint8_t> input = make_input(kind, size, engine);
        CHECK_ROUND_TRIP(input, name_of(kind));
    }

    // Random histograms straight into the builder, without going through bits.
    for (int trial = 0; trial < 400; ++trial) {
        huffman::histogram freq{};
        const std::size_t symbols = 1u + engine() % 256u;
        const unsigned spread = 1u + engine() % 40u;
        std::uniform_int_distribution<std::uint64_t> pick{1u, std::uint64_t{1} << spread};
        for (std::size_t i = 0; i < symbols; ++i) {
            freq[i] = pick(engine);
        }
        const huffman::code_table table = huffman::build_code_table(freq);
        CHECK_EQ(table.symbols, symbols);
        CHECK_TABLE(table, "random histogram");
    }
}

void test_container_round_trip()
{
    std::mt19937 engine{42};

    std::vector<std::vector<std::uint8_t>> datasets;
    datasets.emplace_back();
    datasets.emplace_back(1, std::uint8_t{0});
    datasets.emplace_back(1, std::uint8_t{'x'});
    datasets.emplace_back(4096, std::uint8_t{'z'});
    datasets.emplace_back();
    datasets.back().assign({0, 0, 0, 1, 1, 2});
    {
        std::vector<std::uint8_t> sequence(256);
        for (std::size_t i = 0; i < sequence.size(); ++i) {
            sequence[i] = static_cast<std::uint8_t>(i);
        }
        datasets.push_back(sequence);
    }
    for (const shape kind : {shape::uniform_full, shape::geometric, shape::dominant, shape::power_law,
                             shape::uniform_narrow, shape::alternating}) {
        datasets.push_back(make_input(kind, 70000, engine));
    }

    for (std::size_t index = 0; index < datasets.size(); ++index) {
        const std::vector<std::uint8_t>& input = datasets[index];
        CHECK_ROUND_TRIP(input, "container dataset");

        const std::vector<std::uint8_t> encoded = huffman::compress(input);
        if (input.size() >= 4096) {
            CHECK(encoded.size() <= input.size() + 600);
        }
    }
}

void test_rejects_bad_input()
{
    const std::vector<std::uint8_t> good = huffman::compress(std::vector<std::uint8_t>{'a', 'b', 'c', 'a'});

    const auto throws = [](std::span<const std::uint8_t> data) {
        try {
            static_cast<void>(huffman::decompress(data));
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };

    CHECK(throws({}));
    CHECK(throws(std::span<const std::uint8_t>{good}.first(8)));
    // One byte short of the complete symbol table.
    CHECK(throws(std::span<const std::uint8_t>{good}.first(huffman::container_header_size(3) - 1)));

    std::vector<std::uint8_t> corrupted = good;
    corrupted[0] = 'X';
    CHECK(throws(corrupted));

    std::vector<std::uint8_t> bad_count = good;
    bad_count[12] = 200; // 200 + 3*256 symbols
    bad_count[13] = 3;
    CHECK(throws(bad_count));

    std::vector<std::uint8_t> bad_length = good;
    bad_length[huffman::container_header_size(3) - 1] = 0; // zero code length
    CHECK(throws(bad_length));

    std::vector<std::uint8_t> bad_max = good;
    bad_max[14] = 0; // max length smaller than the stored lengths
    bad_max[15] = 0;
    CHECK(throws(bad_max));
}

} // namespace

int main()
{
    const std::array<std::pair<const char*, void (*)()>, 11> tests{{
        {"bit stream round trip", test_bit_stream_round_trip},
        {"canonical codes", test_canonical_codes},
        {"code table invariants", test_code_table_invariants},
        {"length limited codes", test_length_limited_codes},
        {"degenerate alphabet", test_degenerate_alphabet},
        {"sizes and padding", test_sizes_and_padding},
        {"decode matches reference", test_decode_matches_reference},
        {"decode rejects incomplete code", test_decode_rejects_incomplete_code},
        {"random fuzz", test_random_fuzz},
        {"container round trip", test_container_round_trip},
        {"rejects bad input", test_rejects_bad_input},
    }};

    for (const auto& [name, function] : tests) {
        const int before = g_failures;
        function();
        std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", name);
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
