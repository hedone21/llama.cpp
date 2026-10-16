// CUDA glue of the ARGUS comparison arms (argus-engine tickets/040; the OpenCL glue is
// ggml-opencl/argus-dagger-cl.h, tickets/031). The logic is in ggml-opencl/argus-dagger.h. Off
// unless an environment variable turns an arm on:
//
//   ARGUS_STATIC_CHUNK=<N>            Static Chunking: drain after every N decode ops
//   ARGUS_GPUSCHED_TABLE=<path>       GPUSched-style scheduling with this op-latency table
//   ARGUS_GPUSCHED_FRAME_MS=<P>       frame period (default 16.6; needs the table)
//   ARGUS_GPUSCHED_TRACE=<path>       per-frame CSV (needs the table)
//   ARGUS_GPUSCHED_PROFILE_OUT=<path> offline mode: time the decode ops per GPU frequency, write a table
//
// Only decode graphs (one token) are touched; prefill graphs run as before.

#pragma once

#include <cuda_runtime.h>

struct ggml_cgraph;
struct ggml_tensor;

namespace argus_dagger {

// Start of a graph on the CUDA backend. Reads the environment on the first call.
// `cuda_graph` = this graph is captured or launched as a CUDA graph (fatal when an arm is on).
void begin_graph(cudaStream_t stream, const ggml_cgraph * graph, bool cuda_graph);
// Around one dispatch of the graph loop (a fused group counts as one).
void before_op(const ggml_tensor * node);
void after_op();
// The backend is being released: close the arms' accounting (summary lines, profile table).
void finish();

// before_op on construction, after_op on destruction: one per loop iteration that dispatches.
struct OpScope {
    explicit OpScope(const ggml_tensor * node) { before_op(node); }
    ~OpScope() { after_op(); }
    OpScope(const OpScope &) = delete;
    OpScope & operator=(const OpScope &) = delete;
};

} // namespace argus_dagger
