#include "transcribe/audio.hpp"

#include <algorithm>
#include <stdexcept>

namespace transcribe {

static_assert(std::atomic<std::size_t>::is_always_lock_free);

AudioRingBuffer::AudioRingBuffer(std::size_t capacity) : storage_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("Audio ring buffer capacity must be positive.");
    }
}

std::size_t AudioRingBuffer::write(std::span<const float> samples) noexcept {
    const std::size_t write_position = write_position_.load(std::memory_order_relaxed);
    const std::size_t read_position = read_position_.load(std::memory_order_acquire);
    const std::size_t used = write_position - read_position;
    const std::size_t count = std::min(samples.size(), storage_.size() - used);
    if (count != 0) {
        const std::size_t first = std::min(count, storage_.size() - write_offset_);
        std::copy_n(samples.data(), first, storage_.data() + write_offset_);
        if (first < count) {
            std::copy_n(samples.data() + first, count - first, storage_.data());
        }
        write_offset_ += count;
        if (write_offset_ >= storage_.size()) write_offset_ -= storage_.size();
    }
    write_position_.store(write_position + count, std::memory_order_release);
    return count;
}

std::size_t AudioRingBuffer::read(std::span<float> samples) noexcept {
    const std::size_t read_position = read_position_.load(std::memory_order_relaxed);
    const std::size_t write_position = write_position_.load(std::memory_order_acquire);
    const std::size_t available = write_position - read_position;
    const std::size_t count = std::min(samples.size(), available);
    if (count != 0) {
        const std::size_t first = std::min(count, storage_.size() - read_offset_);
        std::copy_n(storage_.data() + read_offset_, first, samples.data());
        if (first < count) {
            std::copy_n(storage_.data(), count - first, samples.data() + first);
        }
        read_offset_ += count;
        if (read_offset_ >= storage_.size()) read_offset_ -= storage_.size();
    }
    read_position_.store(read_position + count, std::memory_order_release);
    return count;
}

std::size_t AudioRingBuffer::readable() const noexcept {
    const std::size_t read_position = read_position_.load(std::memory_order_acquire);
    const std::size_t write_position = write_position_.load(std::memory_order_acquire);
    return write_position - read_position;
}

std::size_t AudioRingBuffer::writable() const noexcept {
    const auto write = write_position_.load(std::memory_order_relaxed);
    const auto read = read_position_.load(std::memory_order_acquire);
    return storage_.size() - (write - read);
}

} // namespace transcribe
