// CUDA glue of the Doppeladler-style comparison arm (argus-engine tickets/041; the OpenCL glue is
// ggml-opencl/argus-doppel-cl.h, tickets/032). The logic is in ggml-opencl/argus-doppel.h. Off
// unless ARGUS_DOPPEL_R0 is set; the knobs are the OpenCL glue's:
//
//   ARGUS_DOPPEL_R0=<r>         initial GPU share in (0, 0.995); turns the arm on
//   ARGUS_DOPPEL_GGUF=<path>    the model file (same as -m): the CPU share reads its weights
//   ARGUS_DOPPEL_ADAPTIVE=1     adaptive split (Startup -> Descent -> Lookup, ticket 021 §D3)
//   ARGUS_DOPPEL_THREADS=<n>    CPU threads incl. the dispatch thread (default 8)
//   ARGUS_DOPPEL_NO_FLAGS=1     no done-flags, static split
//   ARGUS_DOPPEL_TRACE=<path>   per-token CSV
//
// Only decode graphs (one token) are split; prefill graphs run as before.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <initializer_list>

struct ggml_backend_cuda_context;

namespace argus_doppel_cuda {

// --- implemented in ggml-cuda.cu (they need its static functions) ---------------------------

// The graph loop would fuse nodes [i, i + ops.size()) (ggml_cuda_can_fuse, no unary ops).
bool gpu_can_fuse(const ggml_cgraph * graph, int i, std::initializer_list<ggml_op> ops);
// The graph loop would run this MUL_MAT through the fused mul_mat_vec_f kernel.
bool gpu_should_fuse_mmvf(const ggml_tensor * mul_mat);
// Run one node with the backend's kernels (ggml_cuda_compute_forward).
bool gpu_forward(ggml_backend_cuda_context & ctx, ggml_tensor * node);

// --- called from ggml-cuda.cu ---------------------------------------------------------------

// Read the environment once. True when the arm is on.
bool init_from_env();
// Start of a graph. `cuda_graph` = captured or launched as a CUDA graph (fatal when the arm is on).
void begin_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, bool cuda_graph);
// At node i of the loop, after the skip checks and before the fusion checks: run the node (or the
// fused group it starts) split. Returns the number of nodes handled, 0 = leave it to the loop.
int dispatch(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i);
// End of a graph.
void end_graph();
// set_tensor of a small input (positions, KV slots).
void note_input(const ggml_tensor * t, const void * data, size_t offset, size_t size);
// The backend is being released: summary line.
void finish();

} // namespace argus_doppel_cuda
