// OpenCL glue of the ARGUS comparison arms (see argus-dagger-cl.h).

#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "argus-dagger-cl.h"
#include "argus-dagger.h"

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

// Current GPU frequency in Hz; readable from the shell user on the S25 (`gpuclk` is not).
const char * FREQ_PATH = "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq";
const char * PING_SRC  = "__kernel void argus_gpusched_ping(void) {}";

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

void cl_ok(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + std::to_string(err));
    }
}

double prof_ms(cl_event ev, cl_profiling_info what) {
    cl_ulong t = 0;
    cl_ok(clGetEventProfilingInfo(ev, what, sizeof(t), &t, nullptr), "clGetEventProfilingInfo");
    return (double) t / 1e6;
}

class ClDevice : public Device {
  public:
    explicit ClDevice(cl_command_queue q) : queue(q), t0(std::chrono::steady_clock::now()) {
        uint32_t mhz;
        if (!read_freq_mhz(&mhz)) {
            throw std::runtime_error(std::string("cannot read the GPU frequency at ") + FREQ_PATH);
        }
        cl_command_queue_properties props = 0;
        cl_ok(clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr), "clGetCommandQueueInfo");
        if (!(props & CL_QUEUE_PROFILING_ENABLE)) {
            throw std::runtime_error("the queue has no profiling");
        }
        cl_context   ctx;
        cl_device_id dev;
        cl_ok(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr), "clGetCommandQueueInfo");
        cl_ok(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr), "clGetCommandQueueInfo");
        cl_int err;
        prog = clCreateProgramWithSource(ctx, 1, &PING_SRC, nullptr, &err);
        cl_ok(err, "clCreateProgramWithSource(ping)");
        cl_ok(clBuildProgram(prog, 1, &dev, "", nullptr, nullptr), "clBuildProgram(ping)");
        kern = clCreateKernel(prog, "argus_gpusched_ping", &err);
        cl_ok(err, "clCreateKernel(ping)");
    }

    ~ClDevice() override {
        if (kern) {
            clReleaseKernel(kern);
        }
        if (prog) {
            clReleaseProgram(prog);
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

    // Submit-to-start of the empty kernel (event profiling), and the host wall time around it.
    Ping ping() override {
        const auto t   = std::chrono::steady_clock::now();
        size_t     gws = 1;
        cl_event   ev;
        cl_ok(clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &gws, nullptr, 0, nullptr, &ev), "enqueue ping");
        cl_ok(clFinish(queue), "clFinish(ping)");
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        const double submit = prof_ms(ev, CL_PROFILING_COMMAND_SUBMIT);
        const double start  = prof_ms(ev, CL_PROFILING_COMMAND_START);
        clReleaseEvent(ev);
        return { start > submit ? start - submit : 0.0, wall };
    }

    uint32_t freq_mhz() override {
        uint32_t mhz = 0;
        read_freq_mhz(&mhz);
        return mhz;
    }

    // Finish the queue, then each op's time = the sum of its kernels' END − START.
    std::vector<double> drain() override {
        cl_ok(clFinish(queue), "clFinish(drain)");
        std::vector<double> ms(pending_ops, 0.0);
        for (const auto & [slot, ev] : events) {
            ms.at(slot) += prof_ms(ev, CL_PROFILING_COMMAND_END) - prof_ms(ev, CL_PROFILING_COMMAND_START);
            clReleaseEvent(ev);
        }
        events.clear();
        pending_ops = 0;
        return ms;
    }

    // Ops enqueued since the last drain, and their kernels' events (op slot, event).
    size_t pending_ops = 0;
    std::vector<std::pair<size_t, cl_event>> events;

  private:
    cl_command_queue queue;
    cl_program       prog = nullptr;
    cl_kernel        kern = nullptr;
    std::chrono::steady_clock::time_point t0;
};

struct State {
    bool                      env_read = false;
    bool                      need_profiling = false;
    std::unique_ptr<Chunker>  chunker;
    std::unique_ptr<Sched>    sched;
    std::unique_ptr<ClDevice> dev;
    cl_command_queue          queue = nullptr;
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

} // namespace

bool init_from_env() {
    State & s = st();
    if (s.env_read) {
        return s.need_profiling;
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
        return false;
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
            return false;
        }
    } catch (const std::exception & e) {
        fatal("GPUSCHED", e.what());
    }
    fprintf(stderr, "%s\n", s.sched->start_line().c_str());
    fflush(stderr);
    s.need_profiling = true;
    return true;
}

bool active() {
    const State & s = st();
    return s.chunker || s.sched;
}

void begin_graph(cl_command_queue queue, const ggml_cgraph * graph) {
    State & s = st();
    if (!s.chunker && !s.sched) {
        return;
    }
    s.queue = queue;
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
                s.dev = std::make_unique<ClDevice>(queue);
            }
            s.sched->begin_token(*s.dev, s.decode_tokens);
        } catch (const std::exception & e) {
            fatal("GPUSCHED", e.what());
        }
    }
    s.decode_tokens++;
}

void before_op(const ggml_tensor * node, const char * fused) {
    State & s = st();
    if (!s.in_decode) {
        return;
    }
    s.graph_ops++;
    if (!s.sched) {
        return;
    }
    std::string label = ggml_op_name(node->op);
    if (fused && *fused) {
        label += std::string("+") + fused;
    }
    label += std::string(":") + node->name;
    try {
        s.sched->before_op(*s.dev, label);
    } catch (const std::exception & e) {
        fatal("GPUSCHED", e.what());
    }
    s.dev->pending_ops++;
    s.op_open = true;
}

void after_op() {
    State & s = st();
    if (!s.in_decode) {
        return;
    }
    if (s.chunker && s.chunker->after_op()) {
        clFinish(s.queue);
    }
    s.op_open = false;
}

bool want_event() {
    const State & s = st();
    return s.op_open;
}

void note_event(cl_event ev) {
    State & s = st();
    if (!s.op_open || !s.dev) {
        return;
    }
    clRetainEvent(ev);
    s.dev->events.push_back({ s.dev->pending_ops - 1, ev });
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
