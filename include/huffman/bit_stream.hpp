#pragma once

// Bit-level input/output stream.
//
// The stream is MSB-first: the first bit written lands in the most significant
// bit of the first output byte. The writer keeps a 64-bit register and emits
// the full 32-bit halves, which keeps the per-symbol cost at a few operations.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace huffman {

/// Returns a mask with the low `len` bits set (len in [0, 32]).
[[nodiscard]] constexpr std::uint32_t low_mask(unsigned len) noexcept
{
    return len >= 32u ? 0xFFFF'FFFFu : ((std::uint32_t{1} << len) - 1u);
}

/// Bytes needed to store `bit_count` bits (the last byte is zero padded).
[[nodiscard]] constexpr std::size_t payload_size(std::size_t bit_count) noexcept
{
    return (bit_count + 7u) / 8u;
}

/// MSB-first bit writer.
///
/// The destination is a raw buffer that the caller has already sized with
/// `payload_size()`: the writer then needs no bounds check and no capacity
/// bookkeeping, which keeps the whole encoder loop in registers.
class bit_writer {
public:
    /// `dst` must have room for `payload_size(bit_count)` bytes.
    explicit bit_writer(std::uint8_t* dst) noexcept
        : begin_{dst}
        , dst_{dst}
    {
    }

    /// Appends the low `len` bits of `value` (len in [0, 32]).
    void put(std::uint32_t value, unsigned len) noexcept
    {
        assert(len <= 32u);
        assert(len == 32u || (value >> len) == 0u);
        acc_ = (acc_ << len) | value;
        bits_ += len;
        if (bits_ >= 32u) {
            write_word();
        }
    }

    /// Writes the trailing partial byte, zero padded. Call exactly once.
    void flush() noexcept
    {
        while (bits_ >= 8u) {
            *dst_++ = static_cast<std::uint8_t>(acc_ >> (bits_ - 8u));
            bits_ -= 8u;
        }
        if (bits_ != 0u) {
            *dst_++ = static_cast<std::uint8_t>(acc_ << (8u - bits_));
            bits_ = 0u;
        }
    }

    [[nodiscard]] std::size_t bytes_written() const noexcept
    {
        return static_cast<std::size_t>(dst_ - begin_);
    }

private:
    void write_word() noexcept
    {
        const auto word = static_cast<std::uint32_t>(acc_ >> (bits_ - 32u));
        // Explicit shifts keep the byte order independent of the host endianness.
        *dst_++ = static_cast<std::uint8_t>(word >> 24u);
        *dst_++ = static_cast<std::uint8_t>(word >> 16u);
        *dst_++ = static_cast<std::uint8_t>(word >> 8u);
        *dst_++ = static_cast<std::uint8_t>(word);
        bits_ -= 32u;
    }

    std::uint8_t* const begin_;
    std::uint8_t* dst_;
    std::uint64_t acc_{0};
    unsigned bits_{0};
};

/// MSB-first bit reader. Bits past the end of the payload read as zero, which
/// lets the decoder run without an extra bounds check per bit.
class bit_reader {
public:
    explicit bit_reader(std::span<const std::uint8_t> data) noexcept
        : ptr_{data.data()}
        , end_{data.data() + data.size()}
    {
    }

    /// Consumes a single bit.
    [[nodiscard]] bool next_bit() noexcept
    {
        if (bits_ == 0u) {
            refill();
            if (bits_ == 0u) {
                return false;
            }
        }
        --bits_;
        return ((buf_ >> bits_) & 1u) != 0u;
    }

    /// Consumes `len` bits (len in [1, 32]).
    [[nodiscard]] std::uint32_t get(unsigned len) noexcept
    {
        assert(len >= 1u && len <= 32u);
        while (bits_ < len && bits_ <= 55u && ptr_ != end_) {
            buf_ = (buf_ << 8u) | *ptr_++;
            bits_ += 8u;
        }
        if (bits_ >= len) {
            bits_ -= len;
            return static_cast<std::uint32_t>(buf_ >> bits_) & low_mask(len);
        }
        // Past the end of the payload: zero-extend.
        const auto value = static_cast<std::uint32_t>(buf_ & low_mask(bits_)) << (len - bits_);
        bits_ = 0u;
        return value;
    }

private:
    void refill() noexcept
    {
        while (bits_ <= 55u && ptr_ != end_) {
            buf_ = (buf_ << 8u) | *ptr_++;
            bits_ += 8u;
        }
    }

    const std::uint8_t* ptr_{nullptr};
    const std::uint8_t* end_{nullptr};
    std::uint64_t buf_{0};
    unsigned bits_{0};
};

} // namespace huffman
