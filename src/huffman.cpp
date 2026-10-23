#include "huffman/huffman.hpp"

#include "huffman/bit_stream.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <utility>

namespace huffman {
namespace {

/// Single source of truth for the container header layout.
namespace header {
constexpr std::uint8_t magic[4] = {'H', 'F', '1', '\0'};
constexpr std::size_t size = 16;          ///< total header size
constexpr std::size_t size_offset = 4;    ///< original size, uint64
constexpr std::size_t symbols_offset = 12; ///< symbol count, uint16
constexpr std::size_t max_len_offset = 14; ///< longest code, uint16
constexpr std::size_t symbol_table = 16;  ///< (byte, length) pairs, 2 bytes each
} // namespace header

void append_u16(std::vector<std::uint8_t>& out, std::uint16_t value)
{
    out.push_back(static_cast<std::uint8_t>(value));
    out.push_back(static_cast<std::uint8_t>(value >> 8u));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64u; shift += 8u) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

[[nodiscard]] std::uint16_t read_u16(std::span<const std::uint8_t> in, std::size_t offset)
{
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(in[offset]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(in[offset + 1u]) << 8u));
}

[[nodiscard]] std::uint64_t read_u64(std::span<const std::uint8_t> in, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8u; ++byte) {
        value |= static_cast<std::uint64_t>(in[offset + byte]) << (8u * byte);
    }
    return value;
}

/// Huffman tree node: leaves carry a symbol, internal nodes carry children.
struct tree_node {
    std::uint64_t freq{0};
    int parent{-1};
    int left{-1};
    int right{-1};
    int symbol{-1};
};

/// Builds the Huffman tree for `freq` and returns the code lengths in `lens`.
/// Symbols with a zero count are left untouched. The value returned is the
/// longest code, which may exceed `max_code_length`.
std::uint8_t build_lengths(const histogram& freq, std::array<std::uint8_t, alphabet_size>& lens)
{
    // Min-heap keyed by (frequency, node index). The index is part of the key so
    // that the tree — and therefore the output — is deterministic.
    using queue_entry = std::pair<std::uint64_t, int>;
    std::priority_queue<queue_entry, std::vector<queue_entry>, std::greater<>> heap;

    std::vector<tree_node> nodes;
    nodes.reserve(2u * alphabet_size);

    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (freq[byte] == 0u) {
            continue;
        }
        const auto index = static_cast<int>(nodes.size());
        nodes.push_back(tree_node{freq[byte], -1, -1, -1, static_cast<int>(byte)});
        heap.emplace(freq[byte], index);
    }

    if (heap.size() < 2u) {
        return 0u;
    }

    while (heap.size() > 1u) {
        const queue_entry first = heap.top();
        heap.pop();
        const queue_entry second = heap.top();
        heap.pop();

        const auto index = static_cast<int>(nodes.size());
        nodes.push_back(tree_node{first.first + second.first, -1, first.second, second.second, -1});
        nodes[static_cast<std::size_t>(first.second)].parent = index;
        nodes[static_cast<std::size_t>(second.second)].parent = index;
        heap.emplace(nodes.back().freq, index);
    }

    // The depth of a leaf is its code length.
    std::uint8_t longest = 0;
    for (const tree_node& node : nodes) {
        if (node.symbol < 0) {
            continue;
        }
        unsigned depth = 0;
        for (int parent = node.parent; parent >= 0;
             parent = nodes[static_cast<std::size_t>(parent)].parent) {
            ++depth;
        }
        const auto len = static_cast<std::uint8_t>(depth);
        lens[static_cast<std::size_t>(node.symbol)] = len;
        longest = std::max(longest, len);
    }
    return longest;
}

/// Complete, uniform code: `ceil(log2(symbols))` bits for every symbol.
void assign_flat_lengths(const histogram& freq, std::array<std::uint8_t, alphabet_size>& lens)
{
    std::size_t symbols = 0;
    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (freq[byte] != 0u) {
            ++symbols;
        }
    }
    unsigned bits = 0;
    while ((std::size_t{1} << bits) < symbols) {
        ++bits;
    }
    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (freq[byte] != 0u) {
            lens[byte] = static_cast<std::uint8_t>(bits);
        }
    }
}

// ---------------------------------------------------------------- decode tables

/// Byte swap used by the look-ahead window.
[[nodiscard]] inline std::uint64_t swap_u64(std::uint64_t value) noexcept
{
#if defined(_MSC_VER)
    return _byteswap_uint64(value);
#else
    return __builtin_bswap64(value);
#endif
}

/// The 64 bits starting at bit `bit`, most significant bit first. Bits past the
/// end of the payload read as zero, so the tail needs no special case.
[[nodiscard]] inline std::uint64_t bit_window(const std::uint8_t* data,
                                              std::size_t size,
                                              std::size_t bit) noexcept
{
    const std::size_t byte = bit >> 3;
    std::uint64_t word = 0;
    if (byte < size) {
        const std::size_t available = size - byte;
        const std::size_t taken = available < sizeof(word) ? available : sizeof(word);
        std::memcpy(&word, data + byte, taken);
    }
    if constexpr (std::endian::native == std::endian::little) {
        word = swap_u64(word);
    }
    return word << (bit & 7u);
}

/// Walks canonical code words one bit at a time, in the `(length, symbol)` order
/// the encoder assigns codes in. `bit` carries the stream position in and out,
/// so a walk can start in the middle of a payload.
///
/// Used for codes longer than the lookup table's root bits, and to reject bit
/// patterns that are not code words at all.
class canonical_walk {
public:
    explicit canonical_walk(const code_table& table)
        : max_len_{table.max_len}
    {
        for (std::size_t len = 1; len <= table.max_len; ++len) {
            for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
                if (table.len[byte] == len) {
                    symbols_[next_++] = static_cast<std::uint8_t>(byte);
                    ++count_[len];
                }
            }
        }
    }

    /// Decodes one symbol from `bit` and advances it. Throws if the bits do not
    /// form a code word; bits past `total_bits` read as zero.
    [[nodiscard]] std::uint8_t next(const std::uint8_t* data,
                                    std::size_t total_bits,
                                    std::size_t& bit) const
    {
        std::uint32_t code = 0;
        std::uint32_t first = 0;
        std::size_t index = 0;
        for (std::size_t len = 1; len <= max_len_; ++len) {
            const bool one = bit < total_bits &&
                             ((data[bit >> 3] >> (7u - (bit & 7u))) & 1u) != 0u;
            ++bit;
            code = (code << 1u) | (one ? 1u : 0u);
            if (code - first < count_[len]) {
                return symbols_[index + (code - first)];
            }
            index += count_[len];
            first = (first + count_[len]) << 1u;
        }
        throw std::runtime_error("huffman: payload does not form a valid code word");
    }

private:
    std::array<std::uint8_t, alphabet_size> symbols_{};
    std::array<std::uint32_t, alphabet_size> count_{};
    std::size_t next_{0};
    std::size_t max_len_{0};
};

/// Single-level lookup table for canonical decoding.
///
/// Indexed by the next `root_bits_` bits of the stream, each entry packs as
/// `(length << 8) | byte`. A zero entry means no code of at most `root_bits_`
/// bits starts with that prefix, so the symbol goes through the bit-by-bit
/// walk. For byte data that happens for a small percentage of symbols at most.
class decode_table {
public:
    static constexpr unsigned max_root_bits = 12; ///< 8 KiB, comfortably in L1

    explicit decode_table(const code_table& table)
        : root_bits_{std::min<unsigned>(table.max_len, max_root_bits)}
        , walker_{table}
    {
        code_table coded = table;
        assign_canonical_codes(coded);

        table_.assign(std::size_t{1} << root_bits_, 0);
        for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
            const unsigned len = coded.len[byte];
            if (len == 0u || len > root_bits_) {
                continue;
            }
            const auto entry = static_cast<std::uint16_t>((len << 8) | byte);
            const unsigned shift = root_bits_ - len;
            const unsigned base = coded.code[byte] << shift;
            const unsigned span = 1u << shift;
            for (unsigned k = 0; k < span; ++k) {
                table_[base + k] = entry;
            }
        }
    }

    /// Decodes exactly `symbol_count` symbols into `out`.
    void decode(std::span<const std::uint8_t> payload,
                std::uint64_t symbol_count,
                std::vector<std::uint8_t>& out) const
    {
        out.resize(static_cast<std::size_t>(symbol_count));

        const std::uint8_t* data = payload.data();
        const std::size_t size = payload.size();
        const std::size_t total_bits = size * 8u;
        const std::uint16_t* table = table_.data();
        const unsigned root_shift = 64u - root_bits_;
        std::uint8_t* dst = out.data();
        std::uint8_t* const end = dst + out.size();

        // Left-aligned window: the next bit to decode is bit 63 of `window`,
        // which carries `fill` valid bits (the rest are zero). `pos` is the
        // next byte to load, and the absolute position of the next bit is
        // always `pos * 8 - fill`.
        std::size_t pos = 0;
        std::uint64_t window = 0;
        unsigned fill = 0;

        // Topped up with unaligned 8-byte loads while at least 8 bytes of the
        // payload remain. The splice always leaves `fill >= 57`, which covers
        // the longest code (32 bits) plus the table index (12 bits).
        const auto refill = [&] {
            std::uint64_t word;
            std::memcpy(&word, data + pos, sizeof(word));
            if constexpr (std::endian::native == std::endian::little) {
                word = swap_u64(word);
            }
            window |= word >> fill;
            const unsigned step = (64u - fill) >> 3;
            pos += step;
            fill = 64u - ((64u - fill) & 7u);
        };

        // Realigns the window after a slow walk, which may consume an
        // arbitrary number of bits.
        const auto resync = [&](std::size_t bit) {
            window = bit_window(data, size, bit);
            fill = 64u - static_cast<unsigned>(bit & 7u);
            pos = (bit >> 3) + 8u;
        };

        const auto consume = [&](std::uint16_t entry) {
            *dst++ = static_cast<std::uint8_t>(entry);
            const unsigned len = entry >> 8;
            window <<= len;
            fill -= len;
        };

        // Fast phase: the window can be spliced from a full 8-byte load. Two
        // symbols are taken per pass when the second one is already inside the
        // window, which halves the loop overhead.
        while (dst != end && pos + 8u <= size) {
            if (fill < 57u) {
                refill();
            }
            const std::uint16_t entry = table[window >> root_shift];
            if (entry == 0u) {
                std::size_t bit = pos * 8u - fill;
                *dst++ = walker_.next(data, total_bits, bit);
                resync(bit);
                continue;
            }
            consume(entry);
            if (dst != end && fill >= root_bits_) {
                const std::uint16_t next_entry = table[window >> root_shift];
                if (next_entry != 0u) {
                    consume(next_entry);
                }
            }
        }

        // Tail phase: fewer than 8 bytes left, so top up byte by byte and let
        // bits past the end read as zero.
        while (dst != end) {
            while (fill <= 56u) {
                const std::uint64_t byte = pos < size ? data[pos] : 0u;
                ++pos;
                window |= byte << (56u - fill);
                fill += 8u;
            }
            const std::uint16_t entry = table[window >> root_shift];
            if (entry != 0u) {
                consume(entry);
            } else {
                std::size_t bit = pos * 8u - fill;
                *dst++ = walker_.next(data, total_bits, bit);
                resync(bit);
            }
        }
    }

private:
    unsigned root_bits_{0};
    canonical_walk walker_;
    std::vector<std::uint16_t> table_;
};

} // namespace

void assign_canonical_codes(code_table& table)
{
    // Codes are handed out in (length, symbol) order starting from the smallest
    // code available at each length. The decoder reproduces the same order.
    std::array<std::uint32_t, alphabet_size> length_count{};
    table.max_len = 0;
    for (const std::uint8_t len : table.len) {
        if (len != 0u) {
            ++length_count[len];
            table.max_len = std::max(table.max_len, len);
        }
    }

    std::array<std::uint32_t, alphabet_size> next_code{};
    std::uint32_t code = 0;
    for (std::size_t len = 1; len <= table.max_len; ++len) {
        code = (code + length_count[len - 1u]) << 1u;
        next_code[len] = code;
    }

    for (std::size_t len = 1; len <= table.max_len; ++len) {
        for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
            if (table.len[byte] == len) {
                table.code[byte] = next_code[len]++;
            }
        }
    }
}

std::uint8_t code_table::only_symbol() const noexcept
{
    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (len[byte] != 0u) {
            return static_cast<std::uint8_t>(byte);
        }
    }
    return 0u;
}

histogram count_frequencies(std::span<const std::uint8_t> input) noexcept
{
    histogram freq{};
    for (const std::uint8_t byte : input) {
        ++freq[byte];
    }
    return freq;
}

code_table build_code_table(const histogram& freq)
{
    code_table table{};

    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (freq[byte] != 0u) {
            ++table.symbols;
        }
    }

    // Empty input: nothing to encode.
    if (table.symbols == 0u) {
        return table;
    }

    // A single symbol needs zero bits per symbol. Its length is reported as 1 so
    // that `len != 0` keeps meaning "symbol is present"; the payload stays empty.
    if (table.symbols == 1u) {
        for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
            if (freq[byte] != 0u) {
                table.len[byte] = 1u;
            }
        }
        table.max_len = 1u;
        return table;
    }

    // --- Code lengths -----------------------------------------------------
    // Frequencies are halved while the tree is deeper than the limit. Halving
    // shrinks the dynamic range, so the depth drops; once every count is 1 the
    // tree is balanced and the limit is met, which makes the loop terminate.
    histogram scaled = freq;
    std::array<std::uint8_t, alphabet_size> lens{};
    std::uint8_t longest = 0;
    for (unsigned attempt = 0; attempt < 64u; ++attempt) {
        longest = build_lengths(scaled, lens);
        if (longest <= max_code_length) {
            break;
        }
        for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
            if (scaled[byte] != 0u) {
                scaled[byte] = (scaled[byte] + 1u) / 2u;
            }
        }
    }
    if (longest > max_code_length) {
        assign_flat_lengths(freq, lens); // unreachable in practice, still valid
    }

    table.len = lens;
    assign_canonical_codes(table);
    return table;
}

void encode_payload(const code_table& table,
                    std::span<const std::uint8_t> input,
                    std::size_t bit_count,
                    std::vector<std::uint8_t>& out)
{
    if (table.degenerate()) {
        return; // no bits to write: empty input or a single-symbol alphabet
    }

    const std::size_t start = out.size();
    out.resize(start + payload_size(bit_count));

    bit_writer writer{out.data() + start};
    for (const std::uint8_t byte : input) {
        writer.put(table.code[byte], table.len[byte]);
    }
    writer.flush();

    assert(writer.bytes_written() == payload_size(bit_count));
}

std::size_t encoded_bits(const code_table& table, std::span<const std::uint8_t> input) noexcept
{
    if (table.degenerate()) {
        return 0;
    }
    std::size_t bits = 0;
    for (const std::uint8_t byte : input) {
        bits += table.len[byte];
    }
    return bits;
}

std::size_t encoded_bits(const code_table& table, const histogram& freq) noexcept
{
    if (table.degenerate()) {
        return 0; // zero-length payload, `len` is only a presence marker
    }
    std::size_t bits = 0;
    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        bits += static_cast<std::size_t>(freq[byte]) * table.len[byte];
    }
    return bits;
}

void encode_payload(const code_table& table,
                    std::span<const std::uint8_t> input,
                    std::vector<std::uint8_t>& out)
{
    encode_payload(table, input, encoded_bits(table, input), out);
}

void decode_payload(const code_table& table,
                    std::span<const std::uint8_t> payload,
                    std::uint64_t symbol_count,
                    std::vector<std::uint8_t>& out)
{
    out.clear();

    if (table.symbols == 0u) {
        return; // empty input
    }

    out.reserve(static_cast<std::size_t>(symbol_count));

    if (table.symbols == 1u) {
        out.assign(static_cast<std::size_t>(symbol_count), table.only_symbol());
        return;
    }

    // Table-driven fast path: one lookup for codes up to the root width, a
    // bit-by-bit walk for the rare longer ones and for invalid patterns.
    const decode_table table_for_decode{table};
    table_for_decode.decode(payload, symbol_count, out);
}

std::vector<std::uint8_t> compress(std::span<const std::uint8_t> input)
{
    const histogram freq = count_frequencies(input);
    const code_table table = build_code_table(freq);
    const std::size_t bits = encoded_bits(table, freq);

    std::vector<std::uint8_t> out;
    out.reserve(container_header_size(table.symbols) + payload_size(bits));

    out.insert(out.end(), std::begin(header::magic), std::end(header::magic));
    append_u64(out, input.size());
    append_u16(out, table.symbols);
    append_u16(out, table.max_len);
    for (std::size_t byte = 0; byte < alphabet_size; ++byte) {
        if (table.len[byte] != 0u) {
            out.push_back(static_cast<std::uint8_t>(byte));
            out.push_back(table.len[byte]);
        }
    }

    encode_payload(table, input, bits, out);
    return out;
}

std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> input)
{
    if (input.size() < header::size) {
        throw std::runtime_error("huffman: input is shorter than the header");
    }
    if (!std::equal(std::begin(header::magic), std::end(header::magic), input.begin())) {
        throw std::runtime_error("huffman: bad magic number");
    }

    const std::uint64_t original_size = read_u64(input, header::size_offset);
    const std::uint16_t symbols = read_u16(input, header::symbols_offset);
    if (symbols > alphabet_size) {
        throw std::runtime_error("huffman: symbol count out of range");
    }
    const std::size_t header_size = container_header_size(symbols);
    if (input.size() < header_size) {
        throw std::runtime_error("huffman: truncated symbol table");
    }
    const std::uint16_t max_len = read_u16(input, header::max_len_offset);
    if (symbols > 1u && (max_len == 0u || max_len > max_code_length)) {
        throw std::runtime_error("huffman: code length out of range");
    }
    if (symbols == 0u && original_size != 0u) {
        throw std::runtime_error("huffman: empty symbol table for a non-empty input");
    }

    code_table table{};
    table.symbols = symbols;
    for (std::size_t entry = 0; entry < symbols; ++entry) {
        const std::uint8_t symbol = input[header::symbol_table + 2u * entry];
        const std::uint8_t len = input[header::symbol_table + 2u * entry + 1u];
        if (len == 0u || len > max_len) {
            throw std::runtime_error("huffman: invalid code length in the symbol table");
        }
        table.len[symbol] = len;
    }
    assign_canonical_codes(table);

    std::vector<std::uint8_t> out;
    decode_payload(table, input.subspan(header_size), original_size, out);
    return out;
}

void encode_buffer(std::span<const std::uint8_t> input, std::vector<std::uint8_t>& out)
{
    out.clear();
    const histogram freq = count_frequencies(input);
    const code_table table = build_code_table(freq);
    encode_payload(table, input, encoded_bits(table, freq), out);
}

} // namespace huffman
