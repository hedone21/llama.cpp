// Comparison arms for the ARGUS evaluation (argus-engine tickets/031), on llama.cpp's OpenCL
// decode path:
//
// - Static Chunking (GPUSched PerCom'26 §V-A): wait for the queue after every N decode ops.
// - GPUSched-style scheduling (GPUSched PerCom'26, Kang et al.): per frame of P ms, find the
//   foreground render with a zero-work kernel (GPUPing), compute the GPU time left in the frame,
//   and put only as many decode ops on the GPU as fit in it; sleep for the rest of the frame.
//
// This header is the pure logic: no OpenCL, so it runs in host tests. The OpenCL side (ping
// kernel, events, queue) is behind Device; the glue is argus-dagger-cl.cpp.
//
// The logic is a port of argus-engine `engine/src/gpusched.rs` (tickets/029) with its † choices
// 1-14 and constants unchanged. Differences (tickets/031 † list): an op is one dispatch of the
// graph loop, keyed by its position in the token and its label; op time is the sum of its kernels.

#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace argus_dagger {

// Ping delay above which the GPU counts as rendering (paper §IV-C).
constexpr double PING_THR_MS = 0.5;
// Initial T_overhead (paper §IV-C).
constexpr double OVH0_MS = 4.0;
// Safety margin taken off every budget (paper §IV-C).
constexpr double MARGIN_MS = 0.5;
// 60 Hz frame period, the default P.
constexpr double DEFAULT_FRAME_MS = 16.6;
// WMA window (†).
constexpr size_t WMA_N = 8;
// Samples per op before a new frequency's profile is used (†).
constexpr size_t NEW_FREQ_MIN_SAMPLES = 3;
// Detections behind R̂, the render length (†).
constexpr size_t RHAT_N = 16;
// Next boundary after a detection = estimated render start + P + this (†).
constexpr double ALIGN_EPS_MS = 0.3;
// Gap between the extra pings after an idle ping (†).
constexpr double PROBE_GAP_MS = 0.3;
// No extra ping once the frame's first ping is this old (†).
constexpr double PROBE_WIN_MS = 2.5;
// Extra pings only when a render was detected in this many latest frames (†).
constexpr uint64_t PROBE_RECENT_FRAMES = 8;
// † 15 (tickets/031, added after the first device check): after an oversize frame the next
// boundary keeps the app's frame phase (previous boundary + k·P) instead of re-anchoring on the
// ping, whose render may have waited behind our oversize op.
// Tokens per summary line.
constexpr uint64_t SUMMARY_EVERY = 32;
// Floor for an observed op time, so that no prediction reaches 0.
constexpr double MIN_OP_MS = 1e-4;
// Guard on one chunk's length (never reached with positive predictions and a frame-sized budget).
constexpr size_t MAX_CHUNK_OPS = 1000000;

// span − T_render − T_overhead − 0.5, span = next boundary − ping issue.
double budget_ms(double span_ms, double t_render_ms, double t_overhead_ms);

// T_render when the ping says the GPU was rendering (d > 0.5 ms), else nothing.
std::optional<double> ping_render(double d_ms);

// Next frame boundary. After a detected render (d, R̂): estimated render start
// (ping issue + d − R̂) + P + ALIGN_EPS_MS. After an idle ping: ping end + P.
double next_boundary(double frame_ms, double t_issue_ms, double t_after_ms,
                     std::optional<std::pair<double, double>> render);

// † 15: previous boundary + k·P, the first one after `t_after_ms` (k >= 1).
double held_boundary(double prev_boundary_ms, double frame_ms, double t_after_ms);

struct Greedy {
    size_t n;
    bool   oversize;
};

// Ops in one chunk: the most consecutive ops whose predicted sum is <= budget, and at least one.
// `oversize` is true when even the first op does not fit. pred(k) = prediction of the k-th op.
template <class F>
Greedy greedy_ops(F pred, double budget) {
    double sum = 0.0;
    size_t n   = 0;
    for (size_t k = 0; k < MAX_CHUNK_OPS; ++k) {
        const double p = pred(k);
        if (sum + p > budget) {
            break;
        }
        sum += p;
        n++;
    }
    if (n == 0) {
        return { 1, true };
    }
    return { n, false };
}

// Linearly weighted moving average of the last WMA_N samples (newest weighs most).
class Wma {
  public:
    static Wma seeded(double x);
    void   push(double x);
    size_t len() const { return buf.size(); }
    // Σ k·x_k / Σ k, k = 1 for the oldest sample. Nothing when empty.
    std::optional<double> value() const;

  private:
    std::deque<double> buf;
};

double median(std::vector<double> v);

// ── table file ─────────────────────────────────────────────────────────────
//
// Text, one record per line:
//   version 1
//   ops <n>
//   op <i> <label>                      (n lines, i = 0..n-1)
//   freq <mhz> <samples> <ms_0> ... <ms_{n-1}>
//   # anything                          (ignored; provenance)

struct Table {
    std::vector<std::string>               ops;
    std::map<uint32_t, std::vector<double>> freqs;
    std::map<uint32_t, size_t>              samples;
};

// Throws std::runtime_error on a bad table.
Table parse_table(const std::string & text);
Table load_table(const std::string & path);
std::string table_text(const Table & t, const std::vector<std::string> & provenance);

// Labels go into the table as one word.
std::string sanitize_label(const std::string & s);

// ── predictor ──────────────────────────────────────────────────────────────

class Predictor {
  public:
    explicit Predictor(const Table & t);
    bool     is_valid(uint32_t mhz) const;
    // The valid frequency closest to mhz; on a tie the lower one.
    uint32_t nearest_valid(uint32_t mhz) const;
    // Predicted time of op at mhz (the nearest valid frequency's when mhz has none).
    double   predict(uint32_t mhz, size_t op) const;
    // One observed time of op at mhz.
    void     observe(uint32_t mhz, size_t op, double ms);
    std::vector<uint32_t> freqs() const;

  private:
    struct Profile {
        bool             valid = false;
        std::vector<Wma> ops;
    };
    size_t                      n_ops;
    std::map<uint32_t, Profile> profiles;
};

// ── device ─────────────────────────────────────────────────────────────────

// One ping: submit-to-start delay (events) and host wall time from enqueue to finish.
struct Ping {
    double d_ms;
    double wall_ms;
};

// What the scheduler needs from the GPU side. Errors throw.
class Device {
  public:
    virtual ~Device() = default;
    // Monotonic milliseconds.
    virtual double   now_ms() = 0;
    virtual void     sleep_until_ms(double t_ms) = 0;
    // Enqueue the zero-work kernel and wait for it.
    virtual Ping     ping() = 0;
    // Current GPU frequency, MHz.
    virtual uint32_t freq_mhz() = 0;
    // Finish the queue, then the time (ms) of every op recorded since the last drain, oldest first.
    virtual std::vector<double> drain() = 0;
};

// ── offline profile ────────────────────────────────────────────────────────

class ProfileAcc {
  public:
    // A token's op times, when the frequency did not change across it; else counted as discarded.
    void  add_token(uint32_t mhz_start, uint32_t mhz_end, size_t pos, const std::vector<double> & ms);
    // The table: one median per op for every frequency whose ops all have NEW_FREQ_MIN_SAMPLES
    // samples. `dropped` gets the other frequencies and their fewest sample counts.
    Table to_table(const std::vector<std::string> & ops, std::vector<std::pair<uint32_t, size_t>> * dropped) const;

    uint64_t tokens    = 0;
    uint64_t discarded = 0;
    std::optional<size_t> pos_first;
    size_t   pos_last = 0;

  private:
    std::map<uint32_t, std::vector<std::vector<double>>> samples;
};

// ── scheduler ──────────────────────────────────────────────────────────────

// The frame loop (schedule mode) or op timing per frequency (profile mode).
class Sched {
  public:
    // Schedule mode. `trace` may be null.
    Sched(Table table, std::string table_path, double frame_ms, FILE * trace);
    // Profile mode: the op list is recorded in the first token; the table goes to `out`.
    explicit Sched(std::string out);
    ~Sched();

    std::string start_line() const;
    // Start of a decode token (`pos` = its index, for the profile's provenance).
    void begin_token(Device & dev, size_t pos);
    // Before the op labelled `label` is enqueued. May finish the queue, sleep and ping.
    void before_op(Device & dev, const std::string & label);
    // End of decode: close the open chunk, print the last summary, write the profile table.
    // `dev` null = no device was ever built.
    void finish(Device * dev);

    // Graphs that were not a decode token (prefill), for the summary line.
    uint64_t prefill_graphs = 0;

    uint64_t frames_total() const { return frames_total_; }
    uint64_t oversize_total() const { return oversize_total_; }
    uint64_t tokens() const { return tok; }
    const std::vector<std::string> & ops() const { return ops_; }

  private:
    friend struct SchedPeek;  // host tests

    struct Frame {
        double   t_issue;
        Ping     ping;
        double   t_render;
        double   t_ovh;
        double   budget;
        uint32_t mhz;
        double   next_boundary;
        size_t   n_ops;
        double   pred_sum;
        bool     oversize;
        double   slept_ms;
        uint64_t tok;
        size_t   op0;
        uint32_t pings;
        double   probe_ms;
        double   r_hat;
        bool     hold;  // † 15: boundary kept the phase after an oversize frame
    };
    struct Window {
        size_t frames = 0;
        size_t render = 0;
        std::vector<double> budget, ops, util;
        size_t over  = 0;
        size_t pings = 0;
        std::map<uint32_t, size_t> mhz;
    };

    void   end_token(Device & dev);
    void   next_frame(Device & dev, size_t i);
    void   close_frame(Device & dev, const Frame & f);
    void   summary();
    double unix_ms(Device & dev, double t_ms);

    bool        profile_mode;
    // schedule mode
    std::optional<Predictor> pred;
    double      frame_ms = DEFAULT_FRAME_MS;
    std::string table_path;
    FILE *      trace = nullptr;
    // profile mode
    ProfileAcc  acc;
    std::string out;
    std::optional<std::pair<uint32_t, size_t>> tok_start;

    std::vector<std::string> ops_;
    bool     ops_fixed;
    size_t   op_idx = 0;
    uint64_t tok    = 0;
    size_t   pos    = 0;
    std::optional<Frame> frame;
    size_t   left = 0;
    std::vector<size_t> chunk_ops;
    Wma      ovh = Wma::seeded(OVH0_MS);
    std::deque<double> dets;
    std::optional<uint64_t> last_det;
    uint64_t oversize_total_ = 0;
    uint64_t holds_total_    = 0;
    uint64_t frames_total_   = 0;
    Window   window;
    std::optional<std::pair<double, double>> epoch;
};

// ── Static Chunking ────────────────────────────────────────────────────────

class Chunker {
  public:
    explicit Chunker(size_t n);
    void begin_token();
    // Count one enqueued op; true when the queue must be drained after it.
    bool after_op();
    std::string start_line() const;
    std::string summary() const;

    uint64_t tokens = 0;
    uint64_t prefill_graphs = 0;

  private:
    size_t   n;
    size_t   count  = 0;
    uint64_t ops    = 0;
    uint64_t drains = 0;
};

} // namespace argus_dagger
