#pragma once

#include <cstdlib>

// ARGUS H2O comparison arm. ARGUS_H2O_BUDGET=<ratio> keeps at most ceil(ratio * tokens seen) KV cells:
// half the most recent tokens, half the heavy hitters by attention mass accumulated over every
// query, head and layer so far. KV cells are shared by all layers, so one cell set is kept for all
// of them (layer-shared H2O). Unset or 0 leaves llama.cpp unchanged.
static inline float argus_h2o_budget() {
    static const float budget = [] {
        const char * s = std::getenv("ARGUS_H2O_BUDGET");
        return s ? (float) std::atof(s) : 0.0f;
    }();
    return budget;
}
