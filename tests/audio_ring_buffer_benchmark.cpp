#include "transcribe/audio.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Result {
    std::size_t block_samples{};
    std::size_t capacity_samples{};
    std::size_t samples_per_run{};
    std::vector<double> seconds;
};

Result measure(std::size_t block_samples, std::size_t target_samples,
               std::size_t repetitions) {
    const std::size_t capacity_samples = block_samples * 7 + 13;
    const std::size_t samples_per_run =
        target_samples - target_samples % block_samples;
    std::vector<float> input(block_samples);
    std::vector<float> output(block_samples);
    for (std::size_t i = 0; i < block_samples; ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 128.0f;
    }

    Result result{block_samples, capacity_samples, samples_per_run, {}};
    result.seconds.reserve(repetitions);
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        transcribe::AudioRingBuffer ring(capacity_samples);
        std::size_t processed = 0;
        const auto started = Clock::now();
        while (processed < samples_per_run) {
            const auto count = std::min(block_samples, samples_per_run - processed);
            const auto written = ring.write(std::span<const float>(input).first(count));
            if (written != count) {
                throw std::runtime_error("Ring benchmark write was unexpectedly partial");
            }
            const auto read = ring.read(std::span<float>(output).first(count));
            if (read != count) {
                throw std::runtime_error("Ring benchmark read was unexpectedly partial");
            }
            processed += count;
        }
        const double seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        if (!(seconds > 0.0)) {
            throw std::runtime_error("Ring benchmark timer returned a nonpositive duration");
        }
        if (!std::equal(input.begin(), input.end(), output.begin())) {
            throw std::runtime_error("Ring benchmark changed sample values");
        }
        result.seconds.push_back(seconds);
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
        if (argc > 2 || (argc == 2 && !smoke)) {
            throw std::invalid_argument("Usage: transcriber-audio-ring-benchmark [--smoke]");
        }

        constexpr std::size_t block_sizes[]{32, 64, 128, 256, 512, 1024, 4096};
        const std::size_t target_samples = smoke ? 4096 : 4 * 1024 * 1024;
        const std::size_t repetitions = smoke ? 1 : 5;

        std::cout << std::setprecision(9)
                  << "{\"suite\":\"audio_ring_buffer_v1\",\"samples_per_run_target\":"
                  << target_samples << ",\"repetitions\":" << repetitions
                  << ",\"results\":[";
        for (std::size_t i = 0; i < std::size(block_sizes); ++i) {
            const auto result =
                measure(block_sizes[i], target_samples, repetitions);
            const double seconds = median(result.seconds);
            const double samples_per_second =
                static_cast<double>(result.samples_per_run) / seconds;
            const double copied_mib_per_second =
                2.0 * result.samples_per_run * sizeof(float) / seconds / (1024.0 * 1024.0);
            if (i != 0) std::cout << ',';
            std::cout << "{\"block_samples\":" << result.block_samples
                      << ",\"capacity_samples\":" << result.capacity_samples
                      << ",\"samples_per_run\":" << result.samples_per_run
                      << ",\"median_seconds\":" << seconds
                      << ",\"samples_per_second\":" << samples_per_second
                      << ",\"read_write_mib_per_second\":" << copied_mib_per_second
                      << ",\"seconds\":[";
            for (std::size_t run = 0; run < result.seconds.size(); ++run) {
                if (run != 0) std::cout << ',';
                std::cout << result.seconds[run];
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "transcriber-audio-ring-benchmark: " << error.what() << '\n';
        return 1;
    }
}
