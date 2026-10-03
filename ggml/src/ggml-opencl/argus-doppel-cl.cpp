// OpenCL glue of the Doppeladler-style comparison arm (argus-engine tickets/032). See
// argus-doppel-cl.h for the knobs and argus-doppel.h for the logic.
//
// Per decode layer (ticket 021 §D1, ticket 032 † D1-D4):
//   GPU: attn_norm -> [copy to host staging, flush, input flag] -> Q heads [0,h_g) + bias + RoPE,
//        K/V all heads (unchanged), KV write, attention heads [0,h_g), Wo columns [0,h_g*hd)
//        -> [done flag, flush]
//   CPU: wait input flag -> Q heads [h_g,n) + K/V all heads + bias + RoPE -> host KV append ->
//        attention heads [h_g,n) -> Wo columns [h_g*hd,dim) -> host staging
//   GPU: ffn_inp = Wo + residual (graph) -> [ffn_inp += CPU partial] -> ffn_norm -> [copy, flush,
//        input flag] -> gate/up rows [0,s), swiglu, down columns [0,s) -> [done flag, flush]
//   CPU: gate/up rows [s,ffn), swiglu, down columns [s,ffn) -> host staging
//   GPU: l_out = down + ffn_inp (graph) -> [l_out += CPU partial]

#include "argus-doppel-cl.h"
#include "argus-doppel.h"

#include "ggml-impl.h"
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace argus_doppel_cl {

using namespace argus_doppel;
using clk = std::chrono::steady_clock;

[[noreturn]] static void fatal(const std::string & msg) {
    fprintf(stderr, "[DOPPEL] FATAL %s\n", msg.c_str());
    fflush(stderr);
    std::exit(3);
}

static float ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

// A flag that never rises means a lost command; stop instead of hanging the run.
static const uint64_t MAX_SPINS = 2000000000ull;

enum class Role : uint8_t {
    AttnNorm, MmQ, AddQ, RopeQ, Fa, MmWo, FfnInp, FfnNorm, MmGate, MmUp, Glu, MmDown, LOut,
};

struct LayerNodes {
    ggml_tensor * attn_norm = nullptr; // MUL output of the attention RMS norm
    ggml_tensor * mm_q = nullptr;
    ggml_tensor * add_q = nullptr;
    ggml_tensor * rope_q = nullptr;
    ggml_tensor * mm_k = nullptr;
    ggml_tensor * add_k = nullptr;
    ggml_tensor * rope_k = nullptr;
    ggml_tensor * mm_v = nullptr;
    ggml_tensor * add_v = nullptr;
    ggml_tensor * set_k = nullptr;
    ggml_tensor * set_v = nullptr;
    ggml_tensor * fa = nullptr;
    ggml_tensor * mm_wo = nullptr;
    ggml_tensor * ffn_inp = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * mm_gate = nullptr;
    ggml_tensor * mm_up = nullptr;
    ggml_tensor * glu = nullptr;
    ggml_tensor * mm_down = nullptr;
    ggml_tensor * l_out = nullptr;
};

struct HostWeights {
    const uint16_t * wq = nullptr;
    const uint16_t * wk = nullptr;
    const uint16_t * wv = nullptr;
    const uint16_t * wo = nullptr;
    const uint16_t * wg = nullptr;
    const uint16_t * wu = nullptr;
    const uint16_t * wd = nullptr;
    const float *    bq = nullptr;
    const float *    bk = nullptr;
    const float *    bv = nullptr;
};

struct HostKv {
    std::vector<uint16_t> k; // [n_kv][cap][hd]
    std::vector<uint16_t> v;
    int                   len = 0;
};

struct PendingObs {
    int              layer;
    int              seg;
    int              flag;
    clk::time_point  t0;
    float            t_cpu;
};

struct State {
    // environment
    float       r0       = 1.0f;
    bool        adaptive = false;
    bool        flags    = true;
    int         threads  = 8;
    std::string gguf_path;
    std::string trace_path;

    // backend
    ggml_backend *   backend = nullptr;
    cl_command_queue queue   = nullptr;
    cl_context       ctx     = nullptr;
    bool             installed = false;

    // model geometry (from the first decode graph)
    int    n_layers = 0, n_head = 0, n_kv = 0, hd = 0, dim = 0, ffn = 0, n_dims = 0;
    float  freq_base = 0.0f, scale = 0.0f;
    size_t kv_cap = 0;

    // current graph
    bool                    decode = false;
    const ggml_cgraph *     map_graph = nullptr;
    int                     map_nodes = 0;
    const ggml_tensor *     map_first = nullptr;
    const ggml_tensor *     map_last  = nullptr;
    std::vector<LayerNodes> layers;
    std::unordered_map<const ggml_tensor *, std::pair<int, Role>> roles;
    int                     pos  = -1;
    int                     slot = -1;
    clk::time_point         t_graph;

    // inputs written by set_tensor (positions, KV slots)
    std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> inputs;

    // CPU weights
    void *                   map      = nullptr;
    size_t                   map_size = 0;
    std::vector<HostWeights> hw;

    // host-mapped buffers: inputs [layer][seg][dim], partials [layer][seg][dim], flags [layer][4]
    cl_mem       buf_in = nullptr, buf_out = nullptr, buf_flags = nullptr;
    float *      in_h   = nullptr;
    float *      out_h  = nullptr;
    volatile int * flag_h = nullptr;
    cl_program   prog   = nullptr;
    cl_kernel    k_flag = nullptr;

    // CPU share
    std::unique_ptr<Pool>         pool;
    std::unique_ptr<TpController> ctl;
    std::vector<HostKv>           kv;
    std::vector<float> x, q, k, v, att, wo_part, g, u, act, down_part;
    AttnScratch        scratch;
    std::vector<PendingObs> pending;

    // counters
    uint64_t tokens = 0, prefill_graphs = 0, pos_reads = 0, kv_catchup = 0, kv_full = 0;
    double   token_ms_sum = 0.0;
    FILE *   trace = nullptr;
};

static State * S = nullptr;

static const char * env_str(const char * name) {
    const char * v = getenv(name);
    return (v && *v) ? v : nullptr;
}

bool init_from_env() {
    if (S) {
        return true;
    }
    const char * r0 = env_str("ARGUS_DOPPEL_R0");
    if (!r0) {
        return false;
    }
    if (env_str("ARGUS_STATIC_CHUNK") || env_str("ARGUS_GPUSCHED_TABLE") || env_str("ARGUS_GPUSCHED_PROFILE_OUT")) {
        fatal("ARGUS_DOPPEL_R0 together with a Static/GPUSched arm");
    }
    S        = new State();
    char * e = nullptr;
    S->r0    = strtof(r0, &e);
    if (e == r0 || !(S->r0 > 0.0f) || !(S->r0 < GPU_ONLY_THRESHOLD)) {
        fatal(std::string("ARGUS_DOPPEL_R0 must be in (0, ") + std::to_string(GPU_ONLY_THRESHOLD) + "): " + r0);
    }
    const char * g = env_str("ARGUS_DOPPEL_GGUF");
    if (!g) {
        fatal("ARGUS_DOPPEL_GGUF is required (the -m model file)");
    }
    S->gguf_path = g;
    S->adaptive  = env_str("ARGUS_DOPPEL_ADAPTIVE") && std::string(env_str("ARGUS_DOPPEL_ADAPTIVE")) == "1";
    S->flags     = !(env_str("ARGUS_DOPPEL_NO_FLAGS") && std::string(env_str("ARGUS_DOPPEL_NO_FLAGS")) == "1");
    if (!S->flags && S->adaptive) {
        fatal("ARGUS_DOPPEL_NO_FLAGS=1 needs a static split (ARGUS_DOPPEL_ADAPTIVE unset)");
    }
    if (const char * t = env_str("ARGUS_DOPPEL_THREADS")) {
        S->threads = atoi(t);
        if (S->threads < 1 || S->threads > 64) {
            fatal(std::string("bad ARGUS_DOPPEL_THREADS ") + t);
        }
    }
    if (const char * t = env_str("ARGUS_DOPPEL_TRACE")) {
        S->trace_path = t;
    }
    return true;
}

void note_input(const ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (!S || offset != 0 || size > 16 || (t->type != GGML_TYPE_I32 && t->type != GGML_TYPE_I64)) {
        return;
    }
    std::vector<uint8_t> & v = S->inputs[t];
    v.assign((const uint8_t *) data, (const uint8_t *) data + size);
}

// ---------------------------------------------------------------------------------------------
// Graph map
// ---------------------------------------------------------------------------------------------

static bool is_dispatched(const ggml_tensor * n) {
    return !(ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_VIEW ||
             n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) &&
           (n->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
}

// "blk.<L>.<what>" -> L, what
static bool weight_name(const char * name, int * layer, std::string * what) {
    int  l = -1;
    int  n = 0;
    if (sscanf(name, "blk.%d.%n", &l, &n) != 1 || n == 0) {
        return false;
    }
    *layer = l;
    *what  = name + n;
    return true;
}

// "<base>-<L>" -> L for a node name with the given base.
static int layer_of(const char * name, const char * base) {
    size_t b = strlen(base);
    if (strncmp(name, base, b) != 0 || name[b] != '-') {
        return -1;
    }
    char * e = nullptr;
    long   l = strtol(name + b + 1, &e, 10);
    return (e && *e == '\0') ? (int) l : -1;
}

static ggml_tensor * view_base(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void build_map(ggml_cgraph * g) {
    S->layers.clear();
    S->roles.clear();
    auto lay = [&](int l) -> LayerNodes & {
        if (l < 0 || l > 4096) {
            fatal("bad layer index in graph");
        }
        if ((int) S->layers.size() <= l) {
            S->layers.resize((size_t) l + 1);
        }
        return S->layers[(size_t) l];
    };
    for (int i = 0; i < g->n_nodes; i++) {
        ggml_tensor * n = g->nodes[i];
        if (!is_dispatched(n)) {
            continue;
        }
        int         l = -1;
        std::string what;
        switch (n->op) {
            case GGML_OP_MUL:
                if ((l = layer_of(n->name, "attn_norm")) >= 0) {
                    lay(l).attn_norm = n;
                } else if ((l = layer_of(n->name, "ffn_norm")) >= 0) {
                    lay(l).ffn_norm = n;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (weight_name(n->src[0]->name, &l, &what)) {
                    LayerNodes & L = lay(l);
                    if (what == "attn_q.weight") {
                        L.mm_q = n;
                    } else if (what == "attn_k.weight") {
                        L.mm_k = n;
                    } else if (what == "attn_v.weight") {
                        L.mm_v = n;
                    } else if (what == "attn_output.weight") {
                        L.mm_wo = n;
                    } else if (what == "ffn_gate.weight") {
                        L.mm_gate = n;
                    } else if (what == "ffn_up.weight") {
                        L.mm_up = n;
                    } else if (what == "ffn_down.weight") {
                        L.mm_down = n;
                    } else {
                        fatal(std::string("unexpected matmul weight ") + n->src[0]->name);
                    }
                }
                break;
            case GGML_OP_ADD:
                if (n->src[1] && weight_name(n->src[1]->name, &l, &what)) {
                    if (what == "attn_q.bias") {
                        lay(l).add_q = n;
                    } else if (what == "attn_k.bias") {
                        lay(l).add_k = n;
                    } else if (what == "attn_v.bias") {
                        lay(l).add_v = n;
                    } else {
                        fatal(std::string("unexpected bias ") + n->src[1]->name);
                    }
                } else if ((l = layer_of(n->name, "ffn_inp")) >= 0) {
                    lay(l).ffn_inp = n;
                } else if ((l = layer_of(n->name, "l_out")) >= 0) {
                    lay(l).l_out = n;
                }
                break;
            case GGML_OP_ROPE:
                if ((l = layer_of(n->name, "Qcur")) >= 0) {
                    lay(l).rope_q = n;
                } else if ((l = layer_of(n->name, "Kcur")) >= 0) {
                    lay(l).rope_k = n;
                }
                break;
            case GGML_OP_SET_ROWS: {
                const char * base = view_base(n)->name;
                if (sscanf(base, "cache_k_l%d", &l) == 1) {
                    lay(l).set_k = n;
                } else if (sscanf(base, "cache_v_l%d", &l) == 1) {
                    lay(l).set_v = n;
                }
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT:
                if ((l = layer_of(n->name, "__fattn__")) >= 0) {
                    lay(l).fa = n;
                }
                break;
            case GGML_OP_GLU:
                if ((l = layer_of(n->name, "ffn_swiglu")) >= 0) {
                    lay(l).glu = n;
                }
                break;
            default:
                break;
        }
    }
    if (S->layers.empty()) {
        fatal("decode graph without layers (no attn_norm / matmul nodes found)");
    }
    for (size_t l = 0; l < S->layers.size(); l++) {
        const LayerNodes & L = S->layers[l];
        const ggml_tensor * need[] = {L.attn_norm, L.mm_q, L.add_q, L.rope_q, L.mm_k, L.add_k, L.rope_k,
                                      L.mm_v, L.add_v, L.set_k, L.set_v, L.fa, L.mm_wo, L.ffn_inp,
                                      L.ffn_norm, L.mm_gate, L.mm_up, L.glu, L.mm_down, L.l_out};
        for (const ggml_tensor * t : need) {
            if (!t) {
                fatal("layer " + std::to_string(l) + ": graph does not have the Qwen2 decode layer shape");
            }
        }
        if (!L.glu->src[1]) {
            fatal("fused gate/up GLU (single source) is not supported");
        }
        const std::pair<ggml_tensor *, Role> rs[] = {
            {L.attn_norm, Role::AttnNorm}, {L.mm_q, Role::MmQ},       {L.add_q, Role::AddQ},
            {L.rope_q, Role::RopeQ},       {L.fa, Role::Fa},          {L.mm_wo, Role::MmWo},
            {L.ffn_inp, Role::FfnInp},     {L.ffn_norm, Role::FfnNorm}, {L.mm_gate, Role::MmGate},
            {L.mm_up, Role::MmUp},         {L.glu, Role::Glu},        {L.mm_down, Role::MmDown},
            {L.l_out, Role::LOut},
        };
        for (const auto & r : rs) {
            S->roles[r.first] = {(int) l, r.second};
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Install (first decode graph)
// ---------------------------------------------------------------------------------------------

static void cl_check(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        fatal(std::string(what) + " failed: " + std::to_string(err));
    }
}

static const char * FLAG_SRC =
    "__kernel void argus_doppel_flag(__global volatile int * f, int idx) { atomic_xchg(&f[idx], 1); }\n";

static void * map_buffer(cl_mem buf, size_t size) {
    cl_int err = CL_SUCCESS;
    void * p   = clEnqueueMapBuffer(S->queue, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, nullptr, nullptr, &err);
    cl_check(err, "clEnqueueMapBuffer");
    return p;
}

static cl_mem host_buffer(size_t size) {
    cl_int err = CL_SUCCESS;
    cl_mem b   = clCreateBuffer(S->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
    cl_check(err, "clCreateBuffer(host)");
    return b;
}

static void read_gpu(const ggml_tensor * t, size_t off, size_t size, void * dst) {
    cl_mem mem = nullptr;
    size_t base = 0;
    gpu_tensor_mem(t, &mem, &base);
    cl_check(clEnqueueReadBuffer(S->queue, mem, CL_TRUE, base + off, size, dst, 0, nullptr, nullptr),
             "clEnqueueReadBuffer");
}

static void map_weights(ggml_cgraph * g) {
    (void) g;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context * gc = gguf_init_from_file(S->gguf_path.c_str(), gp);
    if (!gc) {
        fatal("cannot read GGUF " + S->gguf_path);
    }
    int fd = open(S->gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        fatal("cannot open " + S->gguf_path);
    }
    struct stat st;
    fstat(fd, &st);
    S->map_size = (size_t) st.st_size;
    S->map      = mmap(nullptr, S->map_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (S->map == MAP_FAILED) {
        fatal("mmap failed: " + S->gguf_path);
    }
    const size_t data_off = gguf_get_data_offset(gc);
    // Host pointer of a graph weight, checked against the graph tensor's type and size.
    auto host = [&](const ggml_tensor * t, ggml_type type) -> const void * {
        int64_t id = gguf_find_tensor(gc, t->name);
        if (id < 0) {
            fatal(std::string("GGUF has no tensor ") + t->name);
        }
        if (gguf_get_tensor_type(gc, id) != type || t->type != type) {
            fatal(std::string("unexpected type of ") + t->name);
        }
        size_t off  = data_off + gguf_get_tensor_offset(gc, id);
        size_t size = gguf_get_tensor_size(gc, id);
        if (size != ggml_nbytes(t) || off + size > S->map_size) {
            fatal(std::string("size mismatch of ") + t->name);
        }
        return (const uint8_t *) S->map + off;
    };
    S->hw.resize(S->layers.size());
    for (size_t l = 0; l < S->layers.size(); l++) {
        const LayerNodes & L = S->layers[l];
        HostWeights &      w = S->hw[l];
        w.wq = (const uint16_t *) host(L.mm_q->src[0], GGML_TYPE_F16);
        w.wk = (const uint16_t *) host(L.mm_k->src[0], GGML_TYPE_F16);
        w.wv = (const uint16_t *) host(L.mm_v->src[0], GGML_TYPE_F16);
        w.wo = (const uint16_t *) host(L.mm_wo->src[0], GGML_TYPE_F16);
        w.wg = (const uint16_t *) host(L.mm_gate->src[0], GGML_TYPE_F16);
        w.wu = (const uint16_t *) host(L.mm_up->src[0], GGML_TYPE_F16);
        w.wd = (const uint16_t *) host(L.mm_down->src[0], GGML_TYPE_F16);
        w.bq = (const float *) host(L.add_q->src[1], GGML_TYPE_F32);
        w.bk = (const float *) host(L.add_k->src[1], GGML_TYPE_F32);
        w.bv = (const float *) host(L.add_v->src[1], GGML_TYPE_F32);
    }
    gguf_free(gc);
    // † D2: the file must hold the weights the GPU runs (not just the same names and shapes).
    const ggml_tensor * probe[] = {S->layers[0].mm_q->src[0], S->layers[0].mm_down->src[0],
                                   S->layers.back().mm_gate->src[0]};
    const void * hostp[] = {S->hw[0].wq, S->hw[0].wd, S->hw.back().wg};
    for (int i = 0; i < 3; i++) {
        uint8_t dev[64];
        read_gpu(probe[i], 0, sizeof(dev), dev);
        if (memcmp(dev, hostp[i], sizeof(dev)) != 0) {
            fatal(std::string("ARGUS_DOPPEL_GGUF weights differ from the loaded model at ") + probe[i]->name);
        }
    }
}

static void install(ggml_cgraph * g) {
    const LayerNodes & L0 = S->layers[0];
    S->n_layers = (int) S->layers.size();
    S->hd       = (int) L0.rope_q->src[0]->ne[0];
    S->n_head   = (int) L0.rope_q->src[0]->ne[1];
    S->n_kv     = (int) L0.rope_k->src[0]->ne[1];
    S->dim      = (int) L0.mm_q->src[0]->ne[0];
    S->ffn      = (int) L0.mm_gate->src[0]->ne[1];
    if (S->dim != S->n_head * S->hd || L0.mm_q->src[0]->ne[1] != S->dim || L0.mm_wo->src[0]->ne[0] != S->dim ||
        L0.mm_wo->src[0]->ne[1] != S->dim || L0.mm_k->src[0]->ne[1] != S->n_kv * S->hd ||
        L0.mm_down->src[0]->ne[0] != S->ffn || L0.mm_down->src[0]->ne[1] != S->dim || S->n_head % S->n_kv != 0 ||
        S->hd > 256 || S->hd % 8 != 0) {
        fatal("unsupported layer geometry");
    }
    // RoPE (ggml_rope_impl params): n_dims, mode, freq_base, freq_scale, ext_factor, attn_factor.
    const int32_t * rp = (const int32_t *) L0.rope_q->op_params;
    float fb, fs, ef, af;
    memcpy(&fb, rp + 5, 4);
    memcpy(&fs, rp + 6, 4);
    memcpy(&ef, rp + 7, 4);
    memcpy(&af, rp + 8, 4);
    S->n_dims    = rp[1];
    S->freq_base = fb;
    if (rp[2] != GGML_ROPE_TYPE_NEOX || S->n_dims != S->hd || fs != 1.0f || ef != 0.0f || af != 1.0f ||
        L0.rope_q->src[2] != nullptr) {
        fatal("unsupported RoPE (need NeoX, n_dims = head_dim, no scaling, no freq factors)");
    }
    const float * fp = (const float *) L0.fa->op_params;
    S->scale         = fp[0];
    if (fp[1] != 0.0f || fp[2] != 0.0f || L0.fa->src[4] != nullptr) {
        fatal("unsupported attention (ALiBi, softcap or sinks)");
    }
    const ggml_tensor * kc = view_base(L0.set_k);
    const ggml_tensor * vc = view_base(L0.set_v);
    S->kv_cap = (size_t) kc->ne[1];
    if (kc->type != GGML_TYPE_F16 || vc->type != GGML_TYPE_F16 || kc->ne[0] != S->n_kv * S->hd ||
        vc->ne[0] != S->n_kv * S->hd || vc->ne[1] != kc->ne[1] || !ggml_is_contiguous(kc) || !ggml_is_contiguous(vc)) {
        fatal("unsupported KV cache layout (need F16 [n_kv*hd, kv_size], V not transposed)");
    }

    S->queue = gpu_queue(S->backend);
    S->ctx   = gpu_context(S->backend);
    map_weights(g); // reads the GPU copy for the † D2 check: needs the queue

    const size_t io = (size_t) S->n_layers * 2 * S->dim * sizeof(float);
    S->buf_in    = host_buffer(io);
    S->buf_out   = host_buffer(io);
    S->buf_flags = host_buffer((size_t) S->n_layers * 4 * sizeof(int));
    S->in_h      = (float *) map_buffer(S->buf_in, io);
    S->out_h     = (float *) map_buffer(S->buf_out, io);
    S->flag_h    = (volatile int *) map_buffer(S->buf_flags, (size_t) S->n_layers * 4 * sizeof(int));
    for (int i = 0; i < S->n_layers * 4; i++) {
        S->flag_h[i] = 0;
    }
    cl_int        err = CL_SUCCESS;
    cl_device_id  dev = gpu_device(S->backend);
    S->prog           = clCreateProgramWithSource(S->ctx, 1, &FLAG_SRC, nullptr, &err);
    cl_check(err, "clCreateProgramWithSource");
    cl_check(clBuildProgram(S->prog, 1, &dev, "", nullptr, nullptr), "clBuildProgram(flag)");
    S->k_flag = clCreateKernel(S->prog, "argus_doppel_flag", &err);
    cl_check(err, "clCreateKernel(flag)");
    cl_check(clSetKernelArg(S->k_flag, 0, sizeof(cl_mem), &S->buf_flags), "clSetKernelArg(flag)");

    S->pool = std::make_unique<Pool>(S->threads);
    S->ctl  = std::make_unique<TpController>(S->n_layers, S->n_head, S->ffn, S->r0, S->threads, S->adaptive);
    S->kv.resize((size_t) S->n_layers);
    for (HostKv & h : S->kv) {
        h.k.assign((size_t) S->n_kv * S->kv_cap * S->hd, 0);
        h.v.assign((size_t) S->n_kv * S->kv_cap * S->hd, 0);
    }
    S->x.resize((size_t) S->dim);
    S->q.resize((size_t) S->dim);
    S->k.resize((size_t) S->n_kv * S->hd);
    S->v.resize((size_t) S->n_kv * S->hd);
    S->att.resize((size_t) S->dim);
    S->wo_part.resize((size_t) S->dim);
    S->g.resize((size_t) S->ffn);
    S->u.resize((size_t) S->ffn);
    S->act.resize((size_t) S->ffn);
    S->down_part.resize((size_t) S->dim);
    if (!S->trace_path.empty()) {
        S->trace = fopen(S->trace_path.c_str(), "w");
        if (!S->trace) {
            fatal("cannot write " + S->trace_path);
        }
        fprintf(S->trace, "tok,unix_s,pos,token_ms,r_attn,r_ffn,lookup,contention,probe_hits,forced,threads,converged_tok\n");
    }
    S->installed = true;
    fprintf(stderr,
            "[DOPPEL] start layers=%d r0=%.3f adaptive=%d threads=%d flags=%d heads=%d/%d hd=%d dim=%d ffn=%d "
            "kv_cap=%zu gguf=%s\n",
            S->n_layers, S->r0, S->adaptive ? 1 : 0, S->threads, S->flags ? 1 : 0, S->n_head, S->n_kv, S->hd,
            S->dim, S->ffn, S->kv_cap, S->gguf_path.c_str());
}

// ---------------------------------------------------------------------------------------------
// Per graph
// ---------------------------------------------------------------------------------------------

static int64_t input_value(const ggml_tensor * t) {
    auto it = S->inputs.find(t);
    if (it != S->inputs.end() && it->second.size() >= ggml_type_size(t->type)) {
        if (t->type == GGML_TYPE_I32) {
            int32_t v;
            memcpy(&v, it->second.data(), 4);
            return v;
        }
        int64_t v;
        memcpy(&v, it->second.data(), 8);
        return v;
    }
    S->pos_reads++;
    if (t->type == GGML_TYPE_I32) {
        int32_t v = 0;
        read_gpu(t, 0, 4, &v);
        return v;
    }
    int64_t v = 0;
    read_gpu(t, 0, 8, &v);
    return v;
}

// Bring layer l's host KV cache up to `upto` positions from the GPU cache (021 catch_up).
static void catch_up(int l, int upto) {
    HostKv & h  = S->kv[(size_t) l];
    int      lo = h.len > upto ? 0 : h.len;
    if (h.len > upto) {
        S->kv_full++;
    }
    if (lo < upto) {
        const size_t row = (size_t) S->n_kv * S->hd; // halves per token row
        std::vector<uint16_t> tmp((size_t) (upto - lo) * row);
        const ggml_tensor * caches[2] = {view_base(S->layers[(size_t) l].set_k), view_base(S->layers[(size_t) l].set_v)};
        std::vector<uint16_t> * host[2] = {&h.k, &h.v};
        for (int c = 0; c < 2; c++) {
            read_gpu(caches[c], (size_t) lo * caches[c]->nb[1], tmp.size() * 2, tmp.data());
            for (int p = lo; p < upto; p++) {
                for (int kvh = 0; kvh < S->n_kv; kvh++) {
                    memcpy(host[c]->data() + ((size_t) kvh * S->kv_cap + p) * S->hd,
                           tmp.data() + (size_t) (p - lo) * row + (size_t) kvh * S->hd, (size_t) S->hd * 2);
                }
            }
        }
        S->kv_catchup++;
    }
    h.len = upto;
}

void begin_graph(ggml_backend * backend, ggml_cgraph * g) {
    if (!S) {
        return;
    }
    S->decode = false;
    const ggml_tensor * first = nullptr;
    for (int i = 0; i < g->n_nodes; i++) {
        if (is_dispatched(g->nodes[i])) {
            first = g->nodes[i];
            break;
        }
    }
    if (!first) {
        return;
    }
    if (first->ne[1] != 1) {
        S->prefill_graphs++;
        // A prompt may follow a cleared or replaced cache: re-read the host KV from the GPU at the
        // next decode graph instead of trusting rows of an earlier sequence.
        for (HostKv & h : S->kv) {
            h.len = 0;
        }
        return;
    }
    S->backend = backend;
    if (S->map_graph != g || S->map_nodes != g->n_nodes || S->map_first != g->nodes[0] ||
        S->map_last != g->nodes[g->n_nodes - 1]) {
        build_map(g);
        S->map_graph = g;
        S->map_nodes = g->n_nodes;
        S->map_first = g->nodes[0];
        S->map_last  = g->nodes[g->n_nodes - 1];
        if (S->installed && (int) S->layers.size() != S->n_layers) {
            fatal("layer count changed between decode graphs");
        }
    }
    if (!S->installed) {
        install(g);
    }
    S->decode  = true;
    S->t_graph = clk::now();
    const LayerNodes & L0 = S->layers[0];
    S->pos  = (int) input_value(L0.rope_q->src[1]);
    S->slot = (int) input_value(L0.set_k->src[1]);
    if (S->slot != S->pos) {
        fatal("KV slot " + std::to_string(S->slot) + " != position " + std::to_string(S->pos) +
              " (single-sequence cache expected)");
    }
    if ((size_t) S->slot >= S->kv_cap) {
        fatal("KV slot outside the cache");
    }
    for (int l = 0; l < S->n_layers; l++) {
        if (S->kv[(size_t) l].len != S->slot) {
            catch_up(l, S->slot);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Flags and observations
// ---------------------------------------------------------------------------------------------

static void enqueue_flag(int idx) {
    cl_check(clSetKernelArg(S->k_flag, 1, sizeof(int), &idx), "clSetKernelArg(flag idx)");
    size_t gws = 1;
    cl_check(clEnqueueNDRangeKernel(S->queue, S->k_flag, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr),
             "enqueue flag");
}

static void flush() {
    cl_check(clFlush(S->queue), "clFlush");
}

static bool flag_up(int idx) {
    return S->flag_h[idx] != 0;
}

static void flag_reset(int idx) {
    S->flag_h[idx] = 0;
}

static void poll_pending() {
    if (S->pending.empty()) {
        return;
    }
    clk::time_point now = clk::now();
    size_t          w   = 0;
    for (size_t i = 0; i < S->pending.size(); i++) {
        PendingObs & p = S->pending[i];
        if (!flag_up(p.flag)) {
            S->pending[w++] = p;
            continue;
        }
        flag_reset(p.flag);
        std::atomic_thread_fence(std::memory_order_acquire);
        S->ctl->observe(p.layer, p.seg, Obs{true, p.t_cpu, ms_between(p.t0, now)});
    }
    S->pending.resize(w);
}

static clk::time_point spin(int idx) {
    for (uint64_t spins = 0;; spins++) {
        if (flag_up(idx)) {
            clk::time_point t = clk::now();
            std::atomic_thread_fence(std::memory_order_acquire);
            flag_reset(idx);
            poll_pending();
            return t;
        }
        poll_pending();
        if (spins == MAX_SPINS) {
            fatal("flag wait timed out (flag " + std::to_string(idx) + ")");
        }
    }
}

static void observe_after_cpu(int l, int seg, int done, clk::time_point t0, clk::time_point t_end,
                              std::optional<float> serial_t_gpu) {
    float t_cpu = ms_between(t0, t_end);
    if (serial_t_gpu) {
        S->ctl->observe(l, seg, Obs{true, t_cpu, *serial_t_gpu});
        return;
    }
    if (flag_up(done)) {
        flag_reset(done);
        S->ctl->observe(l, seg, Obs{false, t_cpu, std::nullopt});
    } else {
        S->pending.push_back({l, seg, done, t0, t_cpu});
    }
}

// ---------------------------------------------------------------------------------------------
// Split nodes
// ---------------------------------------------------------------------------------------------

static int heads_gpu(int l) {
    return SegGeom::attn(S->n_head).split(S->ctl->applied(l, 0));
}

static int rows_gpu(int l) {
    return SegGeom::ffn(S->ffn).split(S->ctl->applied(l, 1));
}

static void forward(ggml_tensor * n) {
    if (!gpu_forward(S->backend, n)) {
        fatal(std::string("op not supported after split: ") + n->name);
    }
}

// Run a MUL_MAT on GPU rows [0, rows) of src0 (output elements [0, rows)).
static void mm_rows(ggml_tensor * node, int rows) {
    ggml_tensor n  = *node;
    ggml_tensor s0 = *node->src[0];
    s0.ne[1]       = rows;
    n.ne[0]        = rows;
    n.src[0]       = &s0;
    forward(&n);
}

// Run a MUL_MAT over the first `cols` columns (K) of src0 and src1: a partial sum of every row.
static void mm_cols(ggml_tensor * node, int cols) {
    ggml_tensor n  = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    s0.ne[0]       = cols;
    s1.ne[0]       = cols;
    n.src[0]       = &s0;
    n.src[1]       = &s1;
    forward(&n);
}

// Element-wise op on the first `count` elements (dst and both sources, dim 0).
static void elementwise_head(ggml_tensor * node, int count) {
    ggml_tensor n  = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    n.ne[0]        = count;
    s0.ne[0]       = count;
    s1.ne[0]       = count;
    n.src[0]       = &s0;
    n.src[1]       = &s1;
    forward(&n);
}

// RoPE of the first `heads` heads (dim 1).
static void rope_heads(ggml_tensor * node, int heads) {
    ggml_tensor n  = *node;
    ggml_tensor s0 = *node->src[0];
    n.ne[1]        = heads;
    s0.ne[1]       = heads;
    n.src[0]       = &s0;
    forward(&n);
}

bool dispatch(ggml_tensor * node) {
    if (!S || !S->decode) {
        return false;
    }
    auto it = S->roles.find(node);
    if (it == S->roles.end()) {
        return false;
    }
    const int l = it->second.first;
    switch (it->second.second) {
        case Role::MmQ:
            mm_rows(node, heads_gpu(l) * S->hd);
            return true;
        case Role::AddQ:
            elementwise_head(node, heads_gpu(l) * S->hd);
            return true;
        case Role::RopeQ:
            rope_heads(node, heads_gpu(l));
            return true;
        case Role::Fa:
            gpu_flash_attn_heads(S->backend, node, heads_gpu(l));
            return true;
        case Role::MmWo:
            mm_cols(node, heads_gpu(l) * S->hd);
            return true;
        case Role::MmGate:
        case Role::MmUp:
            mm_rows(node, rows_gpu(l));
            return true;
        case Role::Glu:
            elementwise_head(node, rows_gpu(l));
            return true;
        case Role::MmDown:
            mm_cols(node, rows_gpu(l));
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------------------------
// CPU shares
// ---------------------------------------------------------------------------------------------

static void cpu_attn(int l, int h_g) {
    const HostWeights & w   = S->hw[(size_t) l];
    const int           hd  = S->hd;
    const int           dim = S->dim;
    const int           q_lo = h_g * hd;
    const int           kvr  = S->n_kv * hd;
    GemvJob jobs[3] = {
        {w.wq + (size_t) q_lo * dim, (size_t) dim, S->q.data() + q_lo, dim - q_lo},
        {w.wk, (size_t) dim, S->k.data(), kvr},
        {w.wv, (size_t) dim, S->v.data(), kvr},
    };
    gemv_f16_multi(*S->pool, S->x.data(), dim, jobs, 3);
    for (int i = q_lo; i < dim; i++) {
        S->q[(size_t) i] += w.bq[i];
    }
    for (int i = 0; i < kvr; i++) {
        S->k[(size_t) i] += w.bk[i];
        S->v[(size_t) i] += w.bv[i];
    }
    rope_neox(S->q.data(), h_g, S->n_head, hd, S->n_dims, (double) S->pos, S->freq_base);
    rope_neox(S->k.data(), 0, S->n_kv, hd, S->n_dims, (double) S->pos, S->freq_base);
    HostKv & kv = S->kv[(size_t) l];
    for (int kvh = 0; kvh < S->n_kv; kvh++) {
        uint16_t * kd = kv.k.data() + ((size_t) kvh * S->kv_cap + S->slot) * hd;
        uint16_t * vd = kv.v.data() + ((size_t) kvh * S->kv_cap + S->slot) * hd;
        for (int d = 0; d < hd; d++) {
            kd[d] = f32_to_f16(S->k[(size_t) kvh * hd + d]);
            vd[d] = f32_to_f16(S->v[(size_t) kvh * hd + d]);
        }
    }
    kv.len = S->slot + 1;
    attention_heads(*S->pool, S->q.data(), kv.k.data(), kv.v.data(), S->kv_cap, S->n_head, S->n_kv, hd, h_g,
                    S->n_head, kv.len, S->scale, S->att.data(), S->scratch);
    GemvJob wo{w.wo + q_lo, (size_t) dim, S->wo_part.data(), dim};
    gemv_f16_multi(*S->pool, S->att.data() + q_lo, dim - q_lo, &wo, 1);
}

static void cpu_ffn(int l, int s) {
    const HostWeights & w    = S->hw[(size_t) l];
    const int           dim  = S->dim;
    const int           rows = S->ffn - s;
    GemvJob jobs[2] = {
        {w.wg + (size_t) s * dim, (size_t) dim, S->g.data(), rows},
        {w.wu + (size_t) s * dim, (size_t) dim, S->u.data(), rows},
    };
    gemv_f16_multi(*S->pool, S->x.data(), dim, jobs, 2);
    silu_mul(S->g.data(), S->u.data(), S->act.data(), rows);
    GemvJob down{w.wd + s, (size_t) S->ffn, S->down_part.data(), dim};
    gemv_f16_multi(*S->pool, S->act.data(), rows, &down, 1);
}

// Entry of a segment: the normed input to host staging, then the input flag in a later
// submission (021 design deviation 2: a flag in the same submission can be seen before the data).
static void entry(ggml_tensor * normed, int l, int seg) {
    cl_mem mem = nullptr;
    size_t off = 0;
    gpu_tensor_mem(normed, &mem, &off);
    const size_t bytes = (size_t) S->dim * sizeof(float);
    cl_check(clEnqueueCopyBuffer(S->queue, mem, S->buf_in, off, ((size_t) l * 2 + seg) * bytes, bytes, 0, nullptr,
                                 nullptr),
             "clEnqueueCopyBuffer(entry)");
    flush();
    enqueue_flag(l * 4 + seg * 2);
}

// End of a segment's GPU share: done flag, flush, then the CPU share and its observation.
static void segment(int l, int seg) {
    const int in_flag   = l * 4 + seg * 2;
    const int done_flag = in_flag + 1;
    if (S->flags) {
        enqueue_flag(done_flag);
    }
    flush();
    const bool      serial = S->flags && S->ctl->serial(l, seg);
    clk::time_point t0     = spin(in_flag);
    memcpy(S->x.data(), S->in_h + ((size_t) l * 2 + seg) * S->dim, (size_t) S->dim * sizeof(float));
    std::optional<float> serial_t_gpu;
    if (serial) {
        serial_t_gpu = ms_between(t0, spin(done_flag));
    }
    clk::time_point t_cpu0 = serial ? clk::now() : t0;
    if (seg == 0) {
        cpu_attn(l, heads_gpu(l));
        memcpy(S->out_h + ((size_t) l * 2) * S->dim, S->wo_part.data(), (size_t) S->dim * sizeof(float));
    } else {
        cpu_ffn(l, rows_gpu(l));
        memcpy(S->out_h + ((size_t) l * 2 + 1) * S->dim, S->down_part.data(), (size_t) S->dim * sizeof(float));
    }
    std::atomic_thread_fence(std::memory_order_release);
    clk::time_point t_end = clk::now();
    if (S->flags) {
        observe_after_cpu(l, seg, done_flag, t_cpu0, t_end, serial_t_gpu);
    }
}

void after(ggml_tensor * out) {
    if (!S || !S->decode) {
        return;
    }
    auto it = S->roles.find(out);
    if (it == S->roles.end()) {
        return;
    }
    const int l = it->second.first;
    switch (it->second.second) {
        case Role::AttnNorm:
            entry(out, l, 0);
            break;
        case Role::MmWo:
            segment(l, 0);
            break;
        case Role::FfnInp:
            gpu_add_buffer(S->backend, out, S->buf_out, ((size_t) l * 2) * S->dim * sizeof(float));
            break;
        case Role::FfnNorm:
            entry(out, l, 1);
            break;
        case Role::MmDown:
            segment(l, 1);
            break;
        case Role::LOut:
            gpu_add_buffer(S->backend, out, S->buf_out, ((size_t) l * 2 + 1) * S->dim * sizeof(float));
            break;
        default:
            break;
    }
}

static void log_tokens() {
    TpStats st = S->ctl->stats();
    fprintf(stderr, "[DOPPEL] tok=%llu r_attn=%.3f r_ffn=%.3f lookup=%d/%d contention=%llu threads=%d\n",
            (unsigned long long) S->tokens, st.r_attn, st.r_ffn, st.lookup, S->n_layers * 2,
            (unsigned long long) st.contention, S->ctl->threads);
}

void end_graph() {
    if (!S || !S->decode) {
        return;
    }
    for (uint64_t spins = 0; !S->pending.empty(); spins++) {
        poll_pending();
        if (spins == MAX_SPINS) {
            fatal("done-flag drain timed out");
        }
    }
    if (std::optional<int> t = S->ctl->end_token()) {
        S->pool->set_active(*t);
    }
    S->tokens++;
    const float token_ms = ms_between(S->t_graph, clk::now());
    S->token_ms_sum += token_ms;
    if (S->trace) {
        TpStats st = S->ctl->stats();
        const double unix_s =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        fprintf(S->trace, "%llu,%.3f,%d,%.3f,%.4f,%.4f,%d,%llu,%llu,%llu,%d,%s\n", (unsigned long long) S->tokens,
                unix_s, S->pos, token_ms, st.r_attn, st.r_ffn, st.lookup, (unsigned long long) st.contention,
                (unsigned long long) S->ctl->probe_hits, (unsigned long long) S->ctl->forced, S->ctl->threads,
                S->ctl->converged_tok ? std::to_string(*S->ctl->converged_tok).c_str() : "");
    }
    if (S->tokens % 32 == 0) {
        log_tokens();
    }
    S->decode = false;
}

void finish() {
    if (!S) {
        return;
    }
    if (S->installed) {
        TpStats st = S->ctl->stats();
        fprintf(stderr,
                "[DOPPEL] summary tokens=%llu layers=%d segs=%d converged_tok=%s forced=%llu contention=%llu "
                "probe_hits=%llu threads=%d\u2192%d r_attn=%.3f r_ffn=%.3f prefill_graphs=%llu pos_reads=%llu "
                "kv_catchup=%llu kv_full=%llu graph_ms=%.2f\n",
                (unsigned long long) S->tokens, S->n_layers, S->n_layers * 2,
                S->ctl->converged_tok ? std::to_string(*S->ctl->converged_tok).c_str() : "none",
                (unsigned long long) S->ctl->forced, (unsigned long long) S->ctl->contention,
                (unsigned long long) S->ctl->probe_hits, S->ctl->threads_initial, S->ctl->threads, st.r_attn,
                st.r_ffn, (unsigned long long) S->prefill_graphs, (unsigned long long) S->pos_reads,
                (unsigned long long) S->kv_catchup, (unsigned long long) S->kv_full,
                S->tokens ? S->token_ms_sum / (double) S->tokens : 0.0);
    } else {
        fprintf(stderr, "[DOPPEL] summary tokens=0 prefill_graphs=%llu (no decode graph)\n",
                (unsigned long long) S->prefill_graphs);
    }
    if (S->trace) {
        fclose(S->trace);
        S->trace = nullptr;
    }
    fflush(stderr);
}

} // namespace argus_doppel_cl
