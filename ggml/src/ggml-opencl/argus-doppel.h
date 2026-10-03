// Doppeladler-style CPU-GPU adaptive tensor-parallel decode arm (argus-engine tickets/032).
//
// Pure logic, host-testable: the split controller (a C++ port of argus-engine
// engine/src/layers/tp_controller.rs, ticket 021 with its design deviation 1), the CPU kernels of
// the CPU share (F16 weights, F32 activations, F32 accumulation) and a small spin thread pool.
// The OpenCL glue lives in argus-doppel-cl.cpp.
//
// stdio only: <iostream> pulls libc++ into libggml-opencl and changes libggml.so's exports
// (ticket 031).

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace argus_doppel {

// ---------------------------------------------------------------------------------------------
// Controller (tp_controller.rs)
// ---------------------------------------------------------------------------------------------

// A share at or above this is "partition off" (tensor_partition.rs GPU_ONLY_THRESHOLD).
constexpr float GPU_ONLY_THRESHOLD = 0.995f;
// Descent step size for the exact-gradient update.
constexpr float DEFAULT_ETA = 0.25f;
// Doppeladler's contention threshold: a segment slower than this x its best time is contended.
constexpr float DEFAULT_CONTENTION_RATIO = 1.2f;
// FFN row quantum (GPU work-group and CPU chunk alignment).
constexpr int FFN_ROW_QUANTUM = 128;
// Consecutive tokens a condition must hold (convergence, contention).
constexpr int STREAK = 3;
// Descent gives up and adopts the current share after this many tokens.
constexpr unsigned MAX_DESCENT_TOKENS = 16;
// Upward probe step cap, in quanta.
constexpr int MAX_PROBE_QUANTA = 8;
// Every this many Lookup tokens, one runs one quantum above the converged share.
constexpr unsigned PROBE_PERIOD = 8;
// Lowest CPU thread count the contention rule may reduce to.
constexpr int MIN_THREADS = 4;
// Threads removed per CPU-contention event.
constexpr int THREAD_STEP = 2;

enum class SegKind { Attn, Ffn };

// Quantization of one segment's GPU share. The share is a quantum index q in [1, n_units - 1]:
// Q heads for ATTN, 128-row blocks for FFN. Both ends stay split while partition is active.
struct SegGeom {
    SegKind kind;
    int     total; // n_heads_q (ATTN) or ffn_hidden (FFN)

    static SegGeom attn(int n_heads_q) { return {SegKind::Attn, n_heads_q}; }
    static SegGeom ffn(int ffn_hidden) { return {SegKind::Ffn, ffn_hidden}; }

    int   unit() const { return kind == SegKind::Attn ? 1 : FFN_ROW_QUANTUM; }
    int   n_units() const { return total / unit(); }
    bool  splittable() const { return n_units() >= 2; }
    int   q_max() const { return n_units() - 1; }
    float share(int q) const { return (float) (q * unit()) / (float) total; }
    int   split(int q) const { return q * unit(); }
    // nullopt = partition off (r >= GPU_ONLY_THRESHOLD, or an axis too short to split).
    std::optional<int> quantize(float r) const;
};

// Quantized ATTN split (GPU heads), nullopt = partition off.
std::optional<int> quantize_attn(float r, int n_heads_q);
// Quantized FFN split (GPU rows), nullopt = partition off.
std::optional<int> quantize_ffn(float r, int ffn_hidden);

struct TpConfig {
    float eta              = DEFAULT_ETA;
    float contention_ratio = DEFAULT_CONTENTION_RATIO;
    // Periodic Lookup probe. Always on in the arm; tests turn it off.
    bool  probe            = true;
};

// One token's measurement of one segment, taken at the share SegState::applied() returned.
struct Obs {
    bool                 waited; // the GPU was still busy when the CPU finished its share
    float                t_cpu;  // ms, CPU share start -> end
    std::optional<float> t_gpu;  // ms, GPU share start -> done-flag; known when waited or serial
    // Segment time: the slower device when known, else the CPU (the GPU finished first).
    float t_seg() const { return (t_gpu && waited) ? std::max(*t_gpu, t_cpu) : t_cpu; }
};

enum class Phase { Startup, Descent, Lookup };

enum class SegEvent { None, Converged, ConvergedForced, ContentionCpu, ContentionGpu, ProbeHit };

struct Hist {
    int   q;
    bool  waited;
    float t_seg;
    float t_cpu;
};

// Controller state of one (layer, segment).
struct SegState {
    SegGeom            geom;
    Phase              phase = Phase::Startup;
    float              r     = 0; // continuous GPU share; q is its quantization
    int                q     = 1;
    unsigned           k_probe = 0;
    std::vector<Hist>  hist; // last observations in Descent, oldest first (<= STREAK)
    std::optional<int> lo;   // Descent bracket: highest quantum seen without a wait
    std::optional<int> hi;   //                  lowest quantum seen with one
    unsigned           descent_tokens = 0;
    int                q_opt          = 1; // converged quantum held in Lookup
    float              t_best         = 0;
    float              t_cpu_best     = 0;
    int                slow_streak    = 0;
    unsigned           since_probe    = 0;
    bool               probing        = false;
    bool               probe_confirm  = false;

    SegState(SegGeom g, float r0);

    int   applied() const { return q; }
    float applied_share() const { return geom.share(q); }
    bool  serial() const { return phase == Phase::Startup; }
    void  restart();
    SegEvent observe(const Obs & obs, const TpConfig & cfg);

  private:
    void     set_r(float r);
    void     set_q(int q);
    void     enter_descent(unsigned k_probe);
    void     enter_lookup(int q_opt, float t_best, float t_cpu_best);
    SegEvent observe_descent(const Obs & obs, const TpConfig & cfg);
    SegEvent observe_lookup(const Obs & obs, const TpConfig & cfg);
    float    t_cpu_median() const;
    std::optional<int> converged_at() const;
    float    best_at(int q) const;
};

struct TpStats {
    float    r_attn     = 0; // mean ATTN GPU share over layers
    float    r_ffn      = 0; // mean FFN GPU share over layers
    int      lookup     = 0; // segments in Lookup
    uint64_t contention = 0;
};

// All segments of a model plus the thread count they share.
struct TpController {
    TpConfig cfg;
    bool     adaptive; // false = static split: observations recorded, shares never move
    std::vector<SegState> segs; // [layer * 2 + seg], seg 0 = ATTN, 1 = FFN
    int      threads;
    int      threads_initial;
    uint32_t tokens = 0;
    std::optional<uint32_t> converged_tok;
    std::vector<bool> converged_once;
    uint64_t forced     = 0;
    uint64_t contention = 0;
    uint64_t probe_hits = 0;
    bool     reduce_pending = false;

    TpController(int n_layers, int n_heads_q, int ffn_hidden, float r0, int threads, bool adaptive,
                 TpConfig cfg = TpConfig());

    int  n_layers() const { return (int) segs.size() / 2; }
    SegState & seg(int layer, int s) { return segs[(size_t) layer * 2 + s]; }
    const SegState & seg(int layer, int s) const { return segs[(size_t) layer * 2 + s]; }
    bool serial(int layer, int s) const { return adaptive && seg(layer, s).serial(); }
    int  applied(int layer, int s) const { return seg(layer, s).applied(); }
    void observe(int layer, int s, const Obs & obs);
    // Close a token: at most one thread step. Returns the new thread count when it changed.
    std::optional<int> end_token();
    TpStats stats() const;
};

// ---------------------------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------------------------

// Spin pool: the calling (dispatch) thread plus `n_total - 1` workers. Workers spin a short time
// after a job, then sleep. set_active() limits how many take part (the contention knob).
class Pool {
  public:
    explicit Pool(int n_total);
    ~Pool();
    Pool(const Pool &) = delete;
    Pool & operator=(const Pool &) = delete;

    int  total() const { return (int) workers.size() + 1; }
    int  active() const { return active_workers.load() + 1; }
    void set_active(int n_total);
    // Run fn(i) for i in [0, n_tasks) on the active threads; returns when all are done.
    void run(int n_tasks, const std::function<void(int)> & fn);

  private:
    void worker(int idx);

    std::vector<std::thread>          workers;
    std::atomic<uint64_t>             gen{0};
    std::atomic<int>                  next{0};
    std::atomic<int>                  done{0};
    std::atomic<int>                  busy{0};
    std::atomic<int>                  sleepers{0};
    std::atomic<int>                  active_workers{0};
    std::atomic<bool>                 stop{false};
    std::atomic<bool>                 open{false};
    int                               n_tasks = 0;
    const std::function<void(int)> *  fn      = nullptr;
    std::mutex                        m;
    std::condition_variable           cv;
};

// ---------------------------------------------------------------------------------------------
// CPU kernels
// ---------------------------------------------------------------------------------------------

float    f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

// y[r] = sum_{i<k} w[r*ld + i] * x[i] for r in [r0, r1). One thread.
void gemv_f16_rows(const float * x, int k, const uint16_t * w, size_t ld, int r0, int r1, float * y);

struct GemvJob {
    const uint16_t * w;    // row 0 of this job
    size_t           ld;   // row stride (elements)
    float *          y;    // rows outputs
    int              rows;
};

// Several GEMVs sharing x[0..k), split into row chunks over the pool.
void gemv_f16_multi(Pool & pool, const float * x, int k, const GemvJob * jobs, int n_jobs);

// NeoX RoPE of heads [h0, h1) of x ([*][hd]) at position pos: pairs (i, i + n_dims/2),
// theta_i = pos * freq_base^(-2i/n_dims).
void rope_neox(float * x, int h0, int h1, int hd, int n_dims, double pos, float freq_base);

// out[i] = silu(g[i]) * u[i]
void silu_mul(const float * g, const float * u, float * out, int n);

// Attention of query heads [h0, h1) over cache positions [0, len). q, out: [n_heads][hd] f32.
// kc, vc: [n_kv][cap][hd] F16. GQA: head h reads KV head h / (n_heads / n_kv).
struct AttnScratch {
    std::vector<float> part; // per (head, chunk): m, l, o[hd]
};
void attention_heads(Pool & pool, const float * q, const uint16_t * kc, const uint16_t * vc,
                     size_t cap, int n_heads, int n_kv, int hd, int h0, int h1, int len,
                     float scale, float * out, AttnScratch & scratch);

// Single-threaded reference of attention_heads for one head (tests).
void attention_ref(const float * q, const uint16_t * kc, const uint16_t * vc, int hd, int len,
                   float scale, float * out);

} // namespace argus_doppel
