// OpenCL glue of the Doppeladler-style comparison arm (argus-engine tickets/032). The logic is in
// argus-doppel.h. Off unless ARGUS_DOPPEL_R0 is set:
//
//   ARGUS_DOPPEL_R0=<r>         initial GPU share in (0, 0.995); turns the arm on
//   ARGUS_DOPPEL_GGUF=<path>    the model file (same as -m): the CPU share reads its weights
//   ARGUS_DOPPEL_ADAPTIVE=1     adaptive split (Startup -> Descent -> Lookup, ticket 021 §D3)
//   ARGUS_DOPPEL_THREADS=<n>    CPU threads incl. the dispatch thread (default 8)
//   ARGUS_DOPPEL_NO_FLAGS=1     no done-flags, static split (flag-cost arm of criterion 5)
//   ARGUS_DOPPEL_TRACE=<path>   per-token CSV
//
// Only decode graphs (one token) are split; prefill graphs run as before.

#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#endif
#include <CL/cl.h>

#include <cstddef>

struct ggml_backend;
struct ggml_cgraph;
struct ggml_tensor;

namespace argus_doppel_cl {

// --- implemented in ggml-opencl.cpp (they need its private types) ---------------------------

// Run one node with the backend's kernels (ggml_cl_compute_forward).
bool gpu_forward(ggml_backend * backend, ggml_tensor * node);
// FLASH_ATTN_EXT of one query token over query heads [0, heads) only. The head count argument of
// the kernel stays the graph's (it selects the KV head); only the work size shrinks.
void gpu_flash_attn_heads(ggml_backend * backend, ggml_tensor * node, int heads);
// dst += buf[offset .. offset + 4 * dst->ne[0]) as F32 (the backend's add kernel).
void gpu_add_buffer(ggml_backend * backend, ggml_tensor * dst, cl_mem buf, size_t offset);
// Device buffer and byte offset of a tensor's data.
void gpu_tensor_mem(const ggml_tensor * t, cl_mem * mem, size_t * offset);
cl_command_queue gpu_queue(ggml_backend * backend);
cl_context       gpu_context(ggml_backend * backend);
cl_device_id     gpu_device(ggml_backend * backend);

// --- called from ggml-opencl.cpp ------------------------------------------------------------

// Read the environment once. True when the arm is on.
bool init_from_env();
// Start of a graph on the OpenCL backend.
void begin_graph(ggml_backend * backend, ggml_cgraph * graph);
// Run `node` split (true) or leave it to the graph loop (false).
bool dispatch(ggml_tensor * node);
// After a dispatch; `out` = the node whose result it produced (the last node of a fusion).
void after(ggml_tensor * out);
// End of a graph.
void end_graph();
// set_tensor of a small input (positions, KV slots).
void note_input(const ggml_tensor * t, const void * data, size_t offset, size_t size);
// The backend is being released: summary line.
void finish();

} // namespace argus_doppel_cl
