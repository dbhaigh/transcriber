#include "transcribe/fingering.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Workload {
    std::string name;
    transcribe::FingeringOptions options;
    std::vector<transcribe::Note> notes;
};

struct Measurement {
    double milliseconds{};
    transcribe::FingeringResult result;
};

std::vector<transcribe::Note> make_scale(std::size_t count) {
    std::vector<transcribe::Note> notes;
    notes.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        notes.push_back({60 + static_cast<int>(index % 25),
                         static_cast<double>(index) * 0.14, 0.11,
                         "benchmark", 0.9});
    }
    return notes;
}

std::vector<transcribe::Note> make_chord_progression(std::size_t chord_count) {
    constexpr std::array<std::array<int, 3>, 4> chords{{
        {60, 64, 67}, {55, 59, 62}, {57, 60, 64}, {53, 57, 60}}};
    std::vector<transcribe::Note> notes;
    notes.reserve(chord_count * chords.front().size());
    for (std::size_t chord = 0; chord < chord_count; ++chord) {
        const double onset = static_cast<double>(chord) * 0.5;
        for (const int pitch : chords[chord % chords.size()]) {
            notes.push_back({pitch, onset, 0.38, "benchmark", 0.9});
        }
    }
    return notes;
}

Workload make_alternate_tuning_workload(std::size_t count) {
    Workload workload;
    workload.name = "7-string-capo";
    workload.options.string_tuning = {64, 59, 55, 50, 45, 40, 35};
    workload.options.capo_fret = 2;
    workload.options.physical_fret_count = 24;
    workload.options.maximum_alternatives = 5;
    workload.notes.reserve(count);
    const auto tuning = transcribe::configured_tuning(workload.options);
    for (std::size_t index = 0; index < count; ++index) {
        const auto string = index % tuning.size();
        const int fret = static_cast<int>((index * 7) % 17);
        workload.notes.push_back({tuning[string] + workload.options.capo_fret + fret,
                                  static_cast<double>(index) * 0.125, 0.1,
                                  "benchmark", 0.9});
    }
    return workload;
}

Workload make_stretch_workload() {
    Workload workload;
    workload.name = "wide-stretch-shape";
    workload.options.maximum_alternatives = 5;
    workload.options.position_overrides = {
        {0, 1, 1, true}, {1, 2, 5, true}, {2, 3, 6, true}};
    workload.notes = {{65, 0.0, 0.5, "benchmark", 0.9},
                      {64, 0.0, 0.5, "benchmark", 0.9},
                      {61, 0.0, 0.5, "benchmark", 0.9}};
    return workload;
}

std::vector<Workload> make_workloads(bool smoke) {
    const std::size_t note_count = smoke ? 18 : 96;
    const std::size_t chord_count = smoke ? 4 : 16;
    std::vector<Workload> workloads;

    Workload scale;
    scale.name = "standard-scale";
    scale.options.maximum_alternatives = 5;
    scale.notes = make_scale(note_count);
    workloads.push_back(std::move(scale));

    workloads.push_back(make_stretch_workload());

    Workload chords;
    chords.name = "chord-progression";
    chords.options.maximum_alternatives = 5;
    chords.notes = make_chord_progression(chord_count);
    workloads.push_back(std::move(chords));

    workloads.push_back(make_alternate_tuning_workload(note_count));
    return workloads;
}

void validate_result(const Workload& workload,
                     const transcribe::FingeringResult& result) {
    if (result.positions.size() != workload.notes.size() ||
        result.notes.size() != workload.notes.size() ||
        !result.fully_playable || !std::isfinite(result.cost.total())) {
        throw std::runtime_error(workload.name +
                                 " produced an incomplete or invalid fingering.");
    }
    transcribe::validate_fingering_assignment(
        workload.notes, result.positions, workload.options);
    for (std::size_t index = 0; index < result.notes.size(); ++index) {
        if (!result.notes[index].playable ||
            result.notes[index].playable_positions.empty()) {
            throw std::runtime_error(workload.name +
                                     " did not return candidates for every note.");
        }
    }
    if (result.alternatives.size() > workload.options.maximum_alternatives ||
        result.cost.unassigned_notes != 0.0 ||
        (workload.name == "wide-stretch-shape" &&
         result.cost.chord_stretch <= 0.0)) {
        throw std::runtime_error(workload.name +
                                 " did not satisfy its benchmark scoring checks.");
    }
    for (const auto& alternative : result.alternatives) {
        transcribe::validate_fingering_assignment(
            workload.notes, alternative.positions, workload.options);
    }
}

bool same_solution(const transcribe::FingeringResult& left,
                   const transcribe::FingeringResult& right) {
    if (left.positions.size() != right.positions.size() ||
        left.alternatives.size() != right.alternatives.size() ||
        left.visited_search_nodes != right.visited_search_nodes ||
        std::abs(left.cost.total() - right.cost.total()) > 1e-9) {
        return false;
    }
    for (std::size_t index = 0; index < left.positions.size(); ++index) {
        if (left.positions[index].string_number != right.positions[index].string_number ||
            left.positions[index].fret != right.positions[index].fret) {
            return false;
        }
    }
    return true;
}

double percentile(std::vector<double> samples, double quantile) {
    std::sort(samples.begin(), samples.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(samples.size()))) - 1;
    return samples[index];
}

int parse_repetitions(const std::string& value) {
    int repetitions{};
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), repetitions);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        repetitions < 1 || repetitions > 50) {
        throw std::invalid_argument("--repetitions must be an integer from 1 to 50.");
    }
    return repetitions;
}

void write_cost(std::ostream& output,
                const transcribe::FingeringCostComponents& cost) {
    output << "{\"total\":" << cost.total()
           << ",\"fret_height\":" << cost.fret_height
           << ",\"preferred_position\":" << cost.preferred_position
           << ",\"hand_movement\":" << cost.hand_movement
           << ",\"string_preference\":" << cost.string_preference
           << ",\"chord_stretch\":" << cost.chord_stretch
           << ",\"unassigned_notes\":" << cost.unassigned_notes
           << ",\"repeated_strings\":" << cost.repeated_strings
           << ",\"string_crossings\":" << cost.string_crossings
           << ",\"awkward_transitions\":" << cost.awkward_transitions << '}';
}

void write_workload(std::ostream& output, const Workload& workload,
                    int repetitions) {
    static_cast<void>(transcribe::assign_fingerings(workload.notes,
                                                    workload.options));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    transcribe::FingeringResult baseline;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = Clock::now();
        auto result =
            transcribe::assign_fingerings(workload.notes, workload.options);
        const auto stop = Clock::now();
        validate_result(workload, result);
        const double milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count();
        if (repetition == 0) {
            baseline = std::move(result);
        } else if (!same_solution(baseline, result)) {
            throw std::runtime_error(workload.name +
                                     " returned nondeterministic results.");
        }
        samples.push_back(milliseconds);
    }

    std::size_t candidate_positions = 0;
    for (const auto& note : baseline.notes) {
        candidate_positions += note.playable_positions.size();
    }
    const double median_ms = percentile(samples, 0.5);
    const double notes_per_second =
        static_cast<double>(workload.notes.size()) / std::max(median_ms, 1e-9) * 1000.0;
    const auto tuning = transcribe::configured_tuning(workload.options);

    output << "{\"name\":\"" << workload.name << "\",\"notes\":"
           << workload.notes.size() << ",\"strings\":" << tuning.size()
           << ",\"tuning\":[";
    for (std::size_t index = 0; index < tuning.size(); ++index) {
        if (index != 0) output << ',';
        output << tuning[index];
    }
    output << "],\"capo_fret\":" << workload.options.capo_fret
           << ",\"candidate_positions\":" << candidate_positions
           << ",\"solutions_returned\":" << baseline.alternatives.size() + 1
           << ",\"fully_playable\":"
           << (baseline.fully_playable ? "true" : "false")
           << ",\"search_truncated\":"
           << (baseline.search_truncated ? "true" : "false")
           << ",\"alternatives_complete\":"
           << (baseline.alternatives_complete ? "true" : "false")
           << ",\"visited_search_nodes\":" << baseline.visited_search_nodes
           << ",\"p50_ms\":" << median_ms
           << ",\"p95_ms\":" << percentile(samples, 0.95)
           << ",\"notes_per_second\":" << notes_per_second
           << ",\"selected_cost\":";
    write_cost(output, baseline.cost);
    output << '}';
}

int run(int argc, char** argv) {
    bool smoke = false;
    bool repetitions_set = false;
    int repetitions = 5;
    std::string output_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help") {
            std::cout << "Usage: transcriber-fingering-benchmark [--smoke] "
                         "[--repetitions 1-50] [--output JSON_PATH]\n";
            return 0;
        }
        if (argument == "--smoke") {
            smoke = true;
        } else if (argument == "--repetitions" && index + 1 < argc) {
            repetitions = parse_repetitions(argv[++index]);
            repetitions_set = true;
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
            if (output_path.empty()) {
                throw std::invalid_argument("--output requires a nonempty path.");
            }
        } else {
            throw std::invalid_argument("Unknown or incomplete benchmark option: " +
                                        argument);
        }
    }
    if (smoke && !repetitions_set) repetitions = 1;

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!output_path.empty()) {
        file.open(output_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("Could not create benchmark report: " +
                                     output_path);
        }
        output = &file;
    }

    *output << std::fixed << std::setprecision(6)
            << "{\"schema_version\":1,\"benchmark\":\"tablature-fingering\","
               "\"repetitions\":"
            << repetitions << ",\"workloads\":[";
    const auto workloads = make_workloads(smoke);
    for (std::size_t index = 0; index < workloads.size(); ++index) {
        if (index != 0) *output << ',';
        write_workload(*output, workloads[index], repetitions);
    }
    *output << "]}\n";
    if (!*output) throw std::runtime_error("Could not write benchmark report.");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "fingering benchmark: " << error.what() << '\n';
        return 1;
    }
}
