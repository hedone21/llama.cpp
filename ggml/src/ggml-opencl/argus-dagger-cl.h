// OpenCL glue of the ARGUS comparison arms (argus-engine tickets/031). The logic is in
// argus-dagger.h. Off unless an environment variable turns an arm on:
//
//   ARGUS_STATIC_CHUNK=<N>            Static Chunking: drain after every N decode ops
//   ARGUS_GPUSCHED_TABLE=<path>       GPUSched-style scheduling with this op-latency table
//   ARGUS_GPUSCHED_FRAME_MS=<P>       frame period (default 16.6; needs the table)
//   ARGUS_GPUSCHED_TRACE=<path>       per-frame CSV (needs the table)
//   ARGUS_GPUSCHED_PROFILE_OUT=<path> offline mode: time the decode ops per GPU frequency, write a table
//
// Only decode graphs (one token) are touched; prefill graphs run as before.

#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#endif
#include <CL/cl.h>

struct ggml_cgraph;
struct ggml_tensor;

namespace argus_dagger {

// Read the environment once. True when the command queue must have profiling.
bool init_from_env();
// Any arm on?
bool active();
// Start of a graph on the OpenCL backend.
void begin_graph(cl_command_queue queue, const ggml_cgraph * graph);
// Around one dispatch of the graph loop. `fused` names the ops fused into `node` ("" if none).
void before_op(const ggml_tensor * node, const char * fused);
void after_op();
// A kernel event of the current op (the glue takes its own reference when it keeps it).
bool want_event();
void note_event(cl_event ev);
// The backend is being released: close the arms' accounting (summary lines, profile table).
void finish();

} // namespace argus_dagger
