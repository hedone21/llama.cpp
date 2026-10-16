// CUDA glue of the ARGUS comparison arms (see argus-dagger-cuda.h). Differences from the OpenCL
// glue (tickets/040 † J1–J6): drain = cudaStreamSynchronize; an op's time = a cudaEvent pair around
// it; the ping's delay = host wall time of an empty kernel and a sync (CUDA has no submit time);
// the GPU frequency comes from the Jetson devfreq node.

#include "argus-dagger-cuda.h"
#include "../ggml-opencl/argus-dagger.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace argus_dagger {
namespace {

// Current GPU frequency in Hz (Jetson AGX Xavier, gv11b).
const char * FREQ_PATH = "/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq";

__global__ void argus_gpusched_ping() {}

[[noreturn]] void fatal(const char * tag, const std::string & msg) {
    fprintf(stderr, "[%s] FATAL %s\n", tag, msg.c_str());
    fflush(stderr);
    std::exit(3);
}

const char * env(const char * name) {
    const char * v = getenv(name);
    return (v && *v) ? v : nullptr;
}

bool read_freq_mhz(uint32_t * mhz) {
    FILE * f = fopen(FREQ_PATH, "r");
    if (!f) {
        return false;
    }
    unsigned long long hz = 0;
    const int n = fscanf(f, "%llu", &hz);
    fclose(f);
    if (n != 1) {
        return false;
    }
    *mhz = (uint32_t) ((hz + 500000ULL) / 1000000ULL);
    return true;
}

void cu_ok(cudaError_t err, const char * what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

class CuDevice : public Device {
  public:
    explicit CuDevice(cudaStream_t s) : stream(s), t0(std::chrono::steady_clock::now()) {
        uint32_t mhz;
        if (!read_freq_mhz(&mhz)) {
            throw std::runtime_error(std::string("cannot read the GPU frequency at ") + FREQ_PATH);
        }
    }

    ~CuDevice() override {
        for (auto & [a, b] : pool) {
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
    }

    double now_ms() override {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void sleep_until_ms(double t_ms) override {
        const double now = now_ms();
        if (t_ms > now) {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(t_ms - now));
        }
    }

    // † J3: the delay is the host wall time from launch to the end of the sync.
    Ping ping() override {
        const auto t = std::chrono::steady_clock::now();
        argus_gpusched_ping<<<1, 1, 0, stream>>>();
        cu_ok(cudaGetLastError(), "launch ping");
        cu_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ping)");
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        return { wall, wall };
    }

    uint32_t freq_mhz() override {
        uint32_t mhz = 0;
        read_freq_mhz(&mhz);
        return mhz;
    }

    // Finish the stream, then each op's time = elapsed between its two events.
    std::vector<double> drain() override {
        cu_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize(drain)");
        std::vector<double> ms(pending_ops, 0.0);
        for (size_t k = 0; k < pending_ops; ++k) {
            float x = 0.0f;
            cu_ok(cudaEventElapsedTime(&x, pool[k].first, pool[k].second), "cudaEventElapsedTime");
            ms[k] = x;
        }
        pending_ops = 0;
        return ms;
    }

    // Start event of a new op (slot = pending_ops), then count it.
    void op_start() {
        if (pending_ops == pool.size()) {
            cudaEvent_t a, b;
            cu_ok(cudaEventCreate(&a), "cudaEventCreate");
            cu_ok(cudaEventCreate(&b), "cudaEventCreate");
            pool.push_back({ a, b });
        }
        cu_ok(cudaEventRecord(pool[pending_ops].first, stream), "cudaEventRecord");
        pending_ops++;
    }

    void op_end() { cu_ok(cudaEventRecord(pool[pending_ops - 1].second, stream), "cudaEventRecord"); }

  private:
    cudaStream_t stream;
    std::chrono::steady_clock::time_point t0;
    size_t pending_ops = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pool;
};

struct State {
    bool                      env_read = false;
    std::unique_ptr<Chunker>  chunker;
    std::unique_ptr<Sched>    sched;
    std::unique_ptr<CuDevice> dev;
    cudaStream_t              stream = nullptr;
    bool                      in_decode = false;
    bool                      op_open = false;
    bool                      seen_decode = false;
    size_t                    decode_tokens = 0;
    // ARGUS_DAGGER_DEBUG: one line per graph (graph kind, first op, ops dispatched).
    bool                      debug = false;
    uint64_t                  graphs = 0;
    size_t                    graph_ops = 0;
};

State & st() {
    static State s;
    return s;
}

// A graph that computes one token: its first dispatched node has one column.
bool is_decode(const ggml_cgraph * g, const ggml_tensor ** first) {
    for (int i = 0; i < g->n_nodes; i++) {
        const ggml_tensor * n = g->nodes[i];
        if (ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_VIEW ||
            n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) {
            continue;
        }
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        *first = n;
        return n->ne[1] == 1;
    }
    *first = nullptr;
    return false;
}

void init_from_env() {
    State & s = st();
    if (s.env_read) {
        return;
    }
    s.env_read = true;
    s.debug    = env("ARGUS_DAGGER_DEBUG") != nullptr;
    const char * sc  = env("ARGUS_STATIC_CHUNK");
    const char * tbl = env("ARGUS_GPUSCHED_TABLE");
    const char * fr  = env("ARGUS_GPUSCHED_FRAME_MS");
    const char * tr  = env("ARGUS_GPUSCHED_TRACE");
    const char * po  = env("ARGUS_GPUSCHED_PROFILE_OUT");
    if (sc && (tbl || fr || tr || po)) {
        fatal("ARGUS", "ARGUS_STATIC_CHUNK and ARGUS_GPUSCHED_* together");
    }
    if (sc) {
        char * end = nullptr;
        const long n = strtol(sc, &end, 10);
        if (!end || *end != '\0' || n < 1) {
            fatal("STATICCHUNK", std::string("ARGUS_STATIC_CHUNK=") + sc + " is not an op count >= 1");
        }
        s.chunker = std::make_unique<Chunker>((size_t) n);
        fprintf(stderr, "%s\n", s.chunker->start_line().c_str());
        fflush(stderr);
        return;
    }
    if ((fr || tr) && !tbl) {
        fatal("GPUSCHED", "ARGUS_GPUSCHED_FRAME_MS / ARGUS_GPUSCHED_TRACE need ARGUS_GPUSCHED_TABLE");
    }
    if (tbl && po) {
        fatal("GPUSCHED", "ARGUS_GPUSCHED_TABLE and ARGUS_GPUSCHED_PROFILE_OUT together");
    }
    try {
        if (tbl) {
            double frame_ms = DEFAULT_FRAME_MS;
            if (fr) {
                char * end = nullptr;
                frame_ms = strtod(fr, &end);
                if (!end || *end != '\0' || !(frame_ms > 0.0)) {
                    fatal("GPUSCHED", std::string("ARGUS_GPUSCHED_FRAME_MS=") + fr + " is not a period > 0");
                }
            }
            FILE * trace = nullptr;
            if (tr) {
                trace = fopen(tr, "w");
                if (!trace) {
                    fatal("GPUSCHED", std::string("cannot create the trace ") + tr);
                }
            }
            s.sched = std::make_unique<Sched>(load_table(tbl), tbl, frame_ms, trace);
        } else if (po) {
            s.sched = std::make_unique<Sched>(std::string(po));
        } else {
            return;
        }
    } catch (const std::exception & e) {
        fatal("GPUSCHED", e.what());
    }
    fprintf(stderr, "%s\n", s.sched->start_line().c_str());
    fflush(stderr);
}

} // namespace

void begin_graph(cudaStream_t stream, const ggml_cgraph * graph, bool cuda_graph) {
    State & s = st();
    init_from_env();
    if (!s.chunker && !s.sched) {
        return;
    }
    if (cuda_graph) {
        fatal("ARGUS", "an arm is on but this graph runs as a CUDA graph");  // † J6
    }
    s.stream = stream;
    const ggml_tensor * first = nullptr;
    s.in_decode = is_decode(graph, &first);
    if (s.debug) {
        fprintf(stderr, "[ARGUS] graph=%llu prev_ops=%zu n_nodes=%d decode=%d first=%s:%s ne1=%lld\n",
                (unsigned long long) s.graphs, s.graph_ops, graph->n_nodes, s.in_decode ? 1 : 0,
                first ? ggml_op_name(first->op) : "-", first ? first->name : "-",
                first ? (long long) first->ne[1] : -1LL);
        s.graphs++;
        s.graph_ops = 0;
    }
    if (!s.in_decode) {
        if (s.chunker) {
            s.chunker->prefill_graphs++;
        }
        if (s.sched) {
            s.sched->prefill_graphs++;
        }
        return;
    }
    s.seen_decode = true;
    if (s.chunker) {
        if (s.chunker->tokens > 0 && s.chunker->tokens % SUMMARY_EVERY == 0) {
            fprintf(stderr, "%s\n", s.chunker->summary().c_str());
            fflush(stderr);
        }
        s.chunker->begin_token();
    }
    if (s.sched) {
        try {
            if (!s.dev) {
                s.dev = std::make_unique<CuDevice>(stream);
            }
            s.sched->begin_token(*s.dev, s.decode_tokens);
        } catch (const std::exception & e) {
            fatal("GPUSCHED", e.what());
        }
    }
    s.decode_tokens++;
}

void before_op(const ggml_tensor * node) {
    State & s = st();
    if (!s.in_decode) {
        return;
    }
    s.graph_ops++;
    if (!s.sched) {
        return;
    }
    // † J5: the label is the first node's; fusion is decided after this hook.
    const std::string label = std::string(ggml_op_name(node->op)) + ":" + node->name;
    try {
        s.sched->before_op(*s.dev, label);
        s.dev->op_start();
    } catch (const std::exception & e) {
        fatal("GPUSCHED", e.what());
    }
    s.op_open = true;
}

void after_op() {
    State & s = st();
    if (!s.in_decode) {
        return;
    }
    if (s.chunker && s.chunker->after_op()) {
        const cudaError_t err = cudaStreamSynchronize(s.stream);
        if (err != cudaSuccess) {
            fatal("STATICCHUNK", std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
        }
    }
    if (s.op_open) {
        try {
            s.dev->op_end();
        } catch (const std::exception & e) {
            fatal("GPUSCHED", e.what());
        }
        s.op_open = false;
    }
}

void finish() {
    State & s = st();
    if (!s.seen_decode) {
        return;
    }
    s.seen_decode = false;
    s.in_decode   = false;
    s.op_open     = false;
    if (s.chunker) {
        fprintf(stderr, "%s\n", s.chunker->summary().c_str());
        fflush(stderr);
    }
    if (s.sched) {
        try {
            s.sched->finish(s.dev.get());
        } catch (const std::exception & e) {
            fatal("GPUSCHED", e.what());
        }
    }
}

} // namespace argus_dagger
