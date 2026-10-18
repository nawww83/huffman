#pragma once

#include <chrono>

namespace huffman {

/// Wall-clock stopwatch based on std::chrono::steady_clock.
class stopwatch {
public:
    using clock = std::chrono::steady_clock;

    stopwatch()
        : start_{clock::now()}
    {
    }

    /// Nanoseconds since construction or the last `restart()`.
    [[nodiscard]] double elapsed_ns() const noexcept
    {
        const auto stop = clock::now();
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_).count());
    }

    void restart() noexcept { start_ = clock::now(); }

private:
    clock::time_point start_;
};

} // namespace huffman
