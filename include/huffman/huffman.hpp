#pragma once

// Canonical Huffman codec.
//
// The pipeline is split into small, individually testable steps:
//
//     count_frequencies()  ->  build_code_table()  ->  encode_payload()
//                                                        decode_payload()
//
// `compress()` / `decompress()` wrap the steps into a self-contained container
// with a compact header, and `encode_buffer()` is the one-call version that the
// benchmark measures.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace huffman {

/// Bytes in the alphabet.
inline constexpr std::size_t alphabet_size = 256;

/// Hard limit on a code word. Frequencies are flattened until the Huffman tree
/// fits, which turns the limit into a postcondition instead of a caveat.
inline constexpr unsigned max_code_length = 32;

/// Symbol counts, indexed by byte value.
using histogram = std::array<std::uint64_t, alphabet_size>;

/// Canonical Huffman code, indexed by byte value.
struct code_table {
    std::array<std::uint32_t, alphabet_size> code{}; ///< right-aligned code bits
    std::array<std::uint8_t, alphabet_size> len{};   ///< 0 == symbol absent
    std::uint16_t symbols{0};                        ///< number of present symbols
    std::uint8_t max_len{0};                         ///< longest code length in bits

    /// True when there is nothing to encode (empty input) or when a single
    /// symbol carries the whole input (a one-symbol Huffman tree emits no bits).
    [[nodiscard]] bool degenerate() const noexcept { return symbols < 2u; }

    /// The only present symbol; valid when `symbols == 1`.
    [[nodiscard]] std::uint8_t only_symbol() const noexcept;
};

/// One pass over the input. Counters are 64-bit, so no rescaling is needed.
[[nodiscard]] histogram count_frequencies(std::span<const std::uint8_t> input) noexcept;

/// Builds the Huffman tree (min-heap, O(M log M)) and assigns canonical codes so
/// that codes are ordered by (length, symbol) — the order the decoder expects.
/// The tree is flattened until every code fits in `max_code_length` bits.
[[nodiscard]] code_table build_code_table(const histogram& freq);

/// Fills `code` (and `max_len`) from `len` using canonical (length, symbol)
/// order. Symbols with `len == 0` are ignored. This is the inverse of the
/// decoder's table construction, so both sides agree on the codes.
void assign_canonical_codes(code_table& table);

/// Number of bits `input` occupies with `table` (one extra pass over the input).
[[nodiscard]] std::size_t encoded_bits(const code_table& table,
                                       std::span<const std::uint8_t> input) noexcept;

/// Same value from an already computed histogram (free).
[[nodiscard]] std::size_t encoded_bits(const code_table& table, const histogram& freq) noexcept;

/// Appends the encoded bit stream to `out` (never clears it). Writes nothing for
/// a degenerate table. `bit_count` must be the exact value from `encoded_bits()`;
/// knowing it up front lets the encoder write into a sized buffer instead of
/// growing a vector while it runs.
void encode_payload(const code_table& table,
                    std::span<const std::uint8_t> input,
                    std::size_t bit_count,
                    std::vector<std::uint8_t>& out);

/// Convenience overload that measures the payload itself.
void encode_payload(const code_table& table,
                    std::span<const std::uint8_t> input,
                    std::vector<std::uint8_t>& out);

/// Decodes exactly `symbol_count` symbols into `out` (cleared first).
/// Throws std::runtime_error if the payload does not form a valid code word.
void decode_payload(const code_table& table,
                    std::span<const std::uint8_t> payload,
                    std::uint64_t symbol_count,
                    std::vector<std::uint8_t>& out);

/// Header + payload, self-contained. Throws std::runtime_error on bad input.
[[nodiscard]] std::vector<std::uint8_t> compress(std::span<const std::uint8_t> input);
[[nodiscard]] std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> input);

/// Histogram + code table + payload in one call (what the benchmark times).
void encode_buffer(std::span<const std::uint8_t> input, std::vector<std::uint8_t>& out);

/// Size of the `compress()` header: magic, original size, symbol count, max
/// length, plus two bytes per present symbol.
[[nodiscard]] constexpr std::size_t container_header_size(std::uint16_t symbols) noexcept
{
    return 16u + 2u * symbols;
}

} // namespace huffman
