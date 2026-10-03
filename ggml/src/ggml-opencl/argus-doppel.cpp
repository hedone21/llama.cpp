// Doppeladler-style adaptive tensor-parallel decode arm: pure logic (argus-engine tickets/032).
// See argus-doppel.h.

#include "argus-doppel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__)
#include <arm_neon.h>
#define ARGUS_DOPPEL_NEON 1
#endif

namespace argus_doppel {

// ---------------------------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------------------------

static const float F32_EPS = std::numeric_limits<float>::epsilon();

std::optional<int> SegGeom::quantize(float r) const {
    if (r >= GPU_ONLY_THRESHOLD || !splittable()) {
        return std::nullopt;
    }
    r = std::max(r, 0.0f);
    long q;
    if (kind == SegKind::Attn) {
        q = std::lround(r * (float) total);
    } else {
        q = (long) (r * (float) total) / unit();
    }
    return (int) std::clamp(q, 1L, (long) q_max());
}

std::optional<int> quantize_attn(float r, int n_heads_q) {
    return SegGeom::attn(n_heads_q).quantize(r);
}

std::optional<int> quantize_ffn(float r, int ffn_hidden) {
    SegGeom g = SegGeom::ffn(ffn_hidden);
    std::optional<int> q = g.quantize(r);
    if (!q) {
        return std::nullopt;
    }
    return g.split(*q);
}

SegState::SegState(SegGeom g, float r0) : geom(g) {
    std::optional<int> q0 = geom.quantize(r0);
    q     = q0 ? *q0 : geom.q_max();
    r     = geom.share(q);
    q_opt = q;
    hist.reserve(STREAK);
}

void SegState::restart() {
    phase         = Phase::Startup;
    probing       = false;
    probe_confirm = false;
    slow_streak   = 0;
    hist.clear();
}

void SegState::set_r(float r_new) {
    float lo_r = geom.share(1);
    float hi_r = geom.share(geom.q_max());
    r = std::clamp(r_new, lo_r, hi_r);
    std::optional<int> q_new = geom.quantize(r);
    q = q_new ? *q_new : geom.q_max();
}

void SegState::set_q(int q_new) {
    q = std::clamp(q_new, 1, geom.q_max());
    r = geom.share(q);
}

void SegState::enter_descent(unsigned k) {
    phase          = Phase::Descent;
    k_probe        = k;
    hist.clear();
    lo.reset();
    hi.reset();
    descent_tokens = 0;
    probing        = false;
    probe_confirm  = false;
    slow_streak    = 0;
}

void SegState::enter_lookup(int q_o, float t_b, float t_cpu_b) {
    phase         = Phase::Lookup;
    q_opt         = q_o;
    t_best        = t_b;
    t_cpu_best    = t_cpu_b;
    slow_streak   = 0;
    since_probe   = 0;
    probing       = false;
    probe_confirm = false;
    set_q(q_o);
}

SegEvent SegState::observe(const Obs & obs, const TpConfig & cfg) {
    switch (phase) {
        case Phase::Startup: {
            // Serial token: both times exact. Balance point from the two measured speeds.
            float r0    = applied_share();
            float t_gpu = std::max(obs.t_gpu ? *obs.t_gpu : obs.t_cpu, F32_EPS);
            float t_cpu = std::max(obs.t_cpu, F32_EPS);
            float v_g   = r0 / t_gpu;
            float v_c   = (1.0f - r0) / t_cpu;
            set_r(v_g / (v_g + v_c));
            enter_descent(0);
            return SegEvent::None;
        }
        case Phase::Descent:
            return observe_descent(obs, cfg);
        case Phase::Lookup:
            return observe_lookup(obs, cfg);
    }
    return SegEvent::None;
}

float SegState::best_at(int qq) const {
    float best = std::numeric_limits<float>::infinity();
    for (const Hist & h : hist) {
        if (h.q == qq) {
            best = std::min(best, h.t_seg);
        }
    }
    return best;
}

float SegState::t_cpu_median() const {
    std::vector<float> v;
    v.reserve(hist.size());
    for (const Hist & h : hist) {
        v.push_back(h.t_cpu);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::optional<int> SegState::converged_at() const {
    if (lo && hi && *hi == *lo + 1) {
        return best_at(*hi) < best_at(*lo) ? *hi : *lo;
    }
    if ((int) hist.size() == STREAK) {
        bool same = true;
        for (const Hist & h : hist) {
            same = same && h.q == hist[0].q;
        }
        if (same) {
            return hist[0].q;
        }
    }
    return std::nullopt;
}

SegEvent SegState::observe_descent(const Obs & obs, const TpConfig & cfg) {
    int q_ran = q;
    descent_tokens++;
    if ((int) hist.size() == STREAK) {
        hist.erase(hist.begin());
    }
    hist.push_back({q_ran, obs.waited, obs.t_seg(), obs.t_cpu});
    // The newest observation wins when noise contradicts the bracket.
    if (obs.waited) {
        hi = q_ran;
        if (lo && !(*lo < q_ran)) {
            lo.reset();
        }
    } else {
        lo = q_ran;
        if (hi && !(*hi > q_ran)) {
            hi.reset();
        }
    }

    // Convergence is judged on what was just observed, before moving again.
    if (std::optional<int> qo = converged_at()) {
        float tb = best_at(*qo);
        if (!std::isfinite(tb)) {
            tb = obs.t_seg();
        }
        enter_lookup(*qo, tb, t_cpu_median());
        return SegEvent::Converged;
    }
    if (descent_tokens >= MAX_DESCENT_TOKENS) {
        float tcb = t_cpu_median();
        enter_lookup(q_ran, obs.t_seg(), tcb);
        return SegEvent::ConvergedForced;
    }

    if (obs.t_gpu && obs.waited) {
        // Exact gradient. GPU slower (t_gpu > t_cpu) -> r falls -> the GPU gets less.
        float t_cpu = obs.t_cpu;
        float t_gpu = *obs.t_gpu;
        float step  = cfg.eta * (t_cpu - t_gpu) / std::max(t_cpu + t_gpu, F32_EPS);
        set_r(r + step);
        k_probe = 0;
    } else {
        // GPU had slack of unknown size -> probe upward, doubling.
        int quanta = (int) std::min<long>(1L << std::min(k_probe, 30u), MAX_PROBE_QUANTA);
        set_q(q + quanta);
        k_probe++;
    }
    // Stay inside the bracket: below the lowest waited quantum, at or above the highest unwaited.
    int q2 = q;
    if (hi) {
        q2 = std::min(q2, std::max(*hi - 1, 1));
    }
    if (lo) {
        q2 = std::max(q2, *lo);
    }
    if (q2 != q) {
        set_q(q2);
    }
    return SegEvent::None;
}

SegEvent SegState::observe_lookup(const Obs & obs, const TpConfig & cfg) {
    if (probing) {
        if (!obs.waited && !probe_confirm) {
            // No wait at one more quantum: run the probe once more before believing it.
            probe_confirm = true;
            return SegEvent::None;
        }
        probing     = false;
        since_probe = 0;
        if (!obs.waited) {
            // Two probe tokens in a row without a wait: slack appeared. The probe token is
            // Descent's first observation, so the next step climbs by 2.
            probe_confirm = false;
            enter_descent(1);
            observe_descent(obs, cfg);
            return SegEvent::ProbeHit;
        }
        probe_confirm = false;
        set_q(q_opt);
        return SegEvent::None;
    }

    float ratio = cfg.contention_ratio;
    if (obs.t_seg() > ratio * t_best) {
        slow_streak++;
    } else {
        slow_streak = 0;
    }
    if (slow_streak >= STREAK) {
        slow_streak = 0;
        if (obs.t_cpu > ratio * t_cpu_best) {
            restart();
            return SegEvent::ContentionCpu;
        }
        // This observation is Descent's first: the gradient step is taken now.
        enter_descent(0);
        observe_descent(obs, cfg);
        return SegEvent::ContentionGpu;
    }

    // The next token is the probe when it completes a period (7 held + 1 probe).
    since_probe++;
    if (cfg.probe && since_probe + 1 >= PROBE_PERIOD && q_opt < geom.q_max()) {
        probing = true;
        set_q(q_opt + 1);
    }
    return SegEvent::None;
}

TpController::TpController(int n_layers, int n_heads_q, int ffn_hidden, float r0, int n_threads,
                           bool adapt, TpConfig c)
    : cfg(c), adaptive(adapt), threads(n_threads), threads_initial(n_threads) {
    segs.reserve((size_t) n_layers * 2);
    for (int l = 0; l < n_layers; l++) {
        segs.emplace_back(SegGeom::attn(n_heads_q), r0);
        segs.emplace_back(SegGeom::ffn(ffn_hidden), r0);
    }
    converged_once.assign((size_t) n_layers * 2, false);
}

void TpController::observe(int layer, int s, const Obs & obs) {
    if (!adaptive) {
        return;
    }
    switch (seg(layer, s).observe(obs, cfg)) {
        case SegEvent::ConvergedForced:
            forced++;
            converged_once[(size_t) layer * 2 + s] = true;
            break;
        case SegEvent::Converged:
            converged_once[(size_t) layer * 2 + s] = true;
            break;
        case SegEvent::ContentionCpu:
            contention++;
            reduce_pending = true;
            break;
        case SegEvent::ContentionGpu:
            contention++;
            break;
        case SegEvent::ProbeHit:
            probe_hits++;
            break;
        case SegEvent::None:
            break;
    }
}

std::optional<int> TpController::end_token() {
    tokens++;
    if (!converged_tok && adaptive &&
        std::all_of(converged_once.begin(), converged_once.end(), [](bool c) { return c; })) {
        converged_tok = tokens;
    }
    if (!reduce_pending) {
        return std::nullopt;
    }
    reduce_pending = false;
    int nxt = std::max(threads - THREAD_STEP, MIN_THREADS);
    // The reduction invalidates every segment's CPU baseline, not only the one that tripped.
    for (SegState & s : segs) {
        s.restart();
    }
    if (nxt == threads) {
        return std::nullopt;
    }
    threads = nxt;
    return nxt;
}

TpStats TpController::stats() const {
    TpStats st;
    int     n = std::max(n_layers(), 1);
    for (int l = 0; l < n_layers(); l++) {
        st.r_attn += seg(l, 0).applied_share();
        st.r_ffn += seg(l, 1).applied_share();
    }
    st.r_attn /= (float) n;
    st.r_ffn /= (float) n;
    for (const SegState & s : segs) {
        st.lookup += s.phase == Phase::Lookup ? 1 : 0;
    }
    st.contention = contention;
    return st;
}

// ---------------------------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------------------------

static inline void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// Spins before a worker sleeps: the ARGUS SpinPool's brief spin (engine/src/thread_pool.rs, 500
// rounds), so idle workers do not burn cores (and heat) while only the GPU works.
static const int SPIN_ROUNDS = 500;

Pool::Pool(int n_total) {
    int nw = std::max(n_total, 1) - 1;
    active_workers.store(nw);
    workers.reserve((size_t) nw);
    for (int i = 0; i < nw; i++) {
        workers.emplace_back([this, i] { worker(i); });
    }
}

Pool::~Pool() {
    stop.store(true);
    {
        std::lock_guard<std::mutex> lk(m);
        gen.fetch_add(1);
    }
    cv.notify_all();
    for (std::thread & t : workers) {
        t.join();
    }
}

void Pool::set_active(int n_total) {
    active_workers.store(std::clamp(n_total - 1, 0, (int) workers.size()));
}

void Pool::worker(int idx) {
    uint64_t seen = 0;
    for (;;) {
        uint64_t g;
        int      spins = 0;
        while ((g = gen.load(std::memory_order_acquire)) == seen) {
            if (stop.load()) {
                return;
            }
            if (++spins < SPIN_ROUNDS) {
                cpu_relax();
                continue;
            }
            std::unique_lock<std::mutex> lk(m);
            sleepers.fetch_add(1);
            cv.wait(lk, [&] { return gen.load() != seen || stop.load(); });
            sleepers.fetch_sub(1);
            spins = 0;
        }
        seen = g;
        if (stop.load()) {
            return;
        }
        if (idx >= active_workers.load()) {
            continue;
        }
        busy.fetch_add(1);
        // Join only an open job: run() closes the job and waits for busy == 0 before it rewrites
        // the job parameters, so a late worker either sees open == false and leaves, or sees a
        // job whose parameters were published before it opened.
        if (!open.load()) {
            busy.fetch_sub(1);
            continue;
        }
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= n_tasks) {
                break;
            }
            (*fn)(i);
            done.fetch_add(1);
        }
        busy.fetch_sub(1);
    }
}

void Pool::run(int nt, const std::function<void(int)> & f) {
    if (nt <= 0) {
        return;
    }
    if (workers.empty() || active_workers.load() == 0 || nt == 1) {
        for (int i = 0; i < nt; i++) {
            f(i);
        }
        return;
    }
    // Workers still inside the previous job must be gone before its parameters change.
    open.store(false);
    while (busy.load() != 0) {
        cpu_relax();
    }
    fn      = &f;
    n_tasks = nt;
    next.store(0);
    done.store(0);
    open.store(true);
    gen.fetch_add(1);
    if (sleepers.load() > 0) {
        std::lock_guard<std::mutex> lk(m);
        cv.notify_all();
    }
    for (;;) {
        int i = next.fetch_add(1);
        if (i >= nt) {
            break;
        }
        f(i);
        done.fetch_add(1);
    }
    while (done.load() < nt) {
        cpu_relax();
    }
    open.store(false);
    while (busy.load() != 0) {
        cpu_relax();
    }
}

// ---------------------------------------------------------------------------------------------
// CPU kernels
// ---------------------------------------------------------------------------------------------

#if !defined(ARGUS_DOPPEL_NEON)
static inline uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline float f32_from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
#endif

float f16_to_f32(uint16_t h) {
#if defined(ARGUS_DOPPEL_NEON)
    __fp16 v;
    std::memcpy(&v, &h, 2);
    return (float) v;
#else
    // FP16 library (Marat Dukhan), as in ggml.
    const uint32_t w           = (uint32_t) h << 16;
    const uint32_t sign        = w & 0x80000000u;
    const uint32_t two_w       = w + w;
    const uint32_t exp_offset  = 0xE0u << 23;
    const float    exp_scale   = 0x1.0p-112f;
    const float    normalized  = f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask  = 126u << 23;
    const float    magic_bias  = 0.5f;
    const float    denormalized = f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t cutoff      = 1u << 27;
    const uint32_t result = sign | (two_w < cutoff ? f32_bits(denormalized) : f32_bits(normalized));
    return f32_from_bits(result);
#endif
}

uint16_t f32_to_f16(float f) {
#if defined(ARGUS_DOPPEL_NEON)
    __fp16 v = (__fp16) f;
    uint16_t h;
    std::memcpy(&h, &v, 2);
    return h;
#else
    const float scale_to_inf  = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float       base          = (std::fabs(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w      = f32_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t       bias   = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }
    base = f32_from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits          = f32_bits(base);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t) ((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
#endif
}

#if defined(ARGUS_DOPPEL_NEON)
static inline float dot_f16_f32(const uint16_t * w, const float * x, int k) {
    float32x4_t a = vdupq_n_f32(0.0f);
    float32x4_t b = vdupq_n_f32(0.0f);
    int         i = 0;
    for (; i + 8 <= k; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + i));
        a = vfmaq_f32(a, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        b = vfmaq_f32(b, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a, b));
    for (; i < k; i++) {
        s += f16_to_f32(w[i]) * x[i];
    }
    return s;
}
#else
static inline float dot_f16_f32(const uint16_t * w, const float * x, int k) {
    float s = 0.0f;
    for (int i = 0; i < k; i++) {
        s += f16_to_f32(w[i]) * x[i];
    }
    return s;
}
#endif

void gemv_f16_rows(const float * x, int k, const uint16_t * w, size_t ld, int r0, int r1, float * y) {
    int r = r0;
#if defined(ARGUS_DOPPEL_NEON)
    const int k8 = k & ~7;
    for (; r + 4 <= r1; r += 4) {
        const uint16_t * w0 = w + (size_t) r * ld;
        const uint16_t * w1 = w0 + ld;
        const uint16_t * w2 = w1 + ld;
        const uint16_t * w3 = w2 + ld;
        float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        for (int i = 0; i < k8; i += 8) {
            const float32x4_t xl = vld1q_f32(x + i);
            const float32x4_t xh = vld1q_f32(x + i + 4);
            float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(w0 + i));
            float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(w1 + i));
            float16x8_t h2 = vreinterpretq_f16_u16(vld1q_u16(w2 + i));
            float16x8_t h3 = vreinterpretq_f16_u16(vld1q_u16(w3 + i));
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), xl);
            b0 = vfmaq_f32(b0, vcvt_high_f32_f16(h0), xh);
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_low_f16(h1)), xl);
            b1 = vfmaq_f32(b1, vcvt_high_f32_f16(h1), xh);
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h2)), xl);
            b2 = vfmaq_f32(b2, vcvt_high_f32_f16(h2), xh);
            a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_low_f16(h3)), xl);
            b3 = vfmaq_f32(b3, vcvt_high_f32_f16(h3), xh);
        }
        float s0 = vaddvq_f32(vaddq_f32(a0, b0));
        float s1 = vaddvq_f32(vaddq_f32(a1, b1));
        float s2 = vaddvq_f32(vaddq_f32(a2, b2));
        float s3 = vaddvq_f32(vaddq_f32(a3, b3));
        for (int i = k8; i < k; i++) {
            s0 += f16_to_f32(w0[i]) * x[i];
            s1 += f16_to_f32(w1[i]) * x[i];
            s2 += f16_to_f32(w2[i]) * x[i];
            s3 += f16_to_f32(w3[i]) * x[i];
        }
        y[r]     = s0;
        y[r + 1] = s1;
        y[r + 2] = s2;
        y[r + 3] = s3;
    }
#endif
    for (; r < r1; r++) {
        y[r] = dot_f16_f32(w + (size_t) r * ld, x, k);
    }
}

void gemv_f16_multi(Pool & pool, const float * x, int k, const GemvJob * jobs, int n_jobs) {
    struct Task {
        int job;
        int r0;
        int r1;
    };
    int total = 0;
    for (int j = 0; j < n_jobs; j++) {
        total += jobs[j].rows;
    }
    if (total == 0) {
        return;
    }
    const int target = pool.active() * 8;
    int       chunk  = (total + target - 1) / target;
    chunk            = std::max(4, (chunk + 3) / 4 * 4);
    std::vector<Task> tasks;
    for (int j = 0; j < n_jobs; j++) {
        for (int r = 0; r < jobs[j].rows; r += chunk) {
            tasks.push_back({j, r, std::min(r + chunk, jobs[j].rows)});
        }
    }
    pool.run((int) tasks.size(), [&](int t) {
        const Task &    tk = tasks[(size_t) t];
        const GemvJob & jb = jobs[tk.job];
        gemv_f16_rows(x, k, jb.w, jb.ld, tk.r0, tk.r1, jb.y);
    });
}

void rope_neox(float * x, int h0, int h1, int hd, int n_dims, double pos, float freq_base) {
    // Same float expression as kernel_rope_neox_f32 (rope.cl): theta = pos * base^(-i0/n_dims).
    const int   half      = n_dims / 2;
    const float inv_ndims = -1.0f / (float) n_dims;
    const float theta_base = (float) pos;
    std::vector<float> c((size_t) half), s((size_t) half);
    for (int ic = 0; ic < half; ic++) {
        const int   i0    = 2 * ic;
        const float theta = theta_base * std::pow(freq_base, inv_ndims * (float) i0);
        c[(size_t) ic]    = std::cos(theta);
        s[(size_t) ic]    = std::sin(theta);
    }
    for (int h = h0; h < h1; h++) {
        float * v = x + (size_t) h * hd;
        for (int ic = 0; ic < half; ic++) {
            const float x0 = v[ic];
            const float x1 = v[ic + half];
            v[ic]          = x0 * c[(size_t) ic] - x1 * s[(size_t) ic];
            v[ic + half]   = x0 * s[(size_t) ic] + x1 * c[(size_t) ic];
        }
    }
}

void silu_mul(const float * g, const float * u, float * out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    }
}

static inline void axpy_f16(float * o, float p, const uint16_t * v, int hd) {
    int i = 0;
#if defined(ARGUS_DOPPEL_NEON)
    const float32x4_t pv = vdupq_n_f32(p);
    for (; i + 8 <= hd; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(v + i));
        vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), vcvt_f32_f16(vget_low_f16(h)), pv));
        vst1q_f32(o + i + 4, vfmaq_f32(vld1q_f32(o + i + 4), vcvt_high_f32_f16(h), pv));
    }
#endif
    for (; i < hd; i++) {
        o[i] += p * f16_to_f32(v[i]);
    }
}

static const int ATTN_CHUNK = 256;
static const int MAX_HD     = 256;

void attention_heads(Pool & pool, const float * q, const uint16_t * kc, const uint16_t * vc,
                     size_t cap, int n_heads, int n_kv, int hd, int h0, int h1, int len,
                     float scale, float * out, AttnScratch & scratch) {
    const int nh = h1 - h0;
    if (nh <= 0) {
        return;
    }
    const int group    = n_heads / n_kv;
    const int n_chunks = std::max(1, (len + ATTN_CHUNK - 1) / ATTN_CHUNK);
    const int stride   = 2 + hd;
    scratch.part.resize((size_t) nh * n_chunks * stride);
    float * part = scratch.part.data();
    pool.run(nh * n_chunks, [&](int t) {
        const int  hh  = t / n_chunks;
        const int  c   = t % n_chunks;
        const int  h   = h0 + hh;
        const int  kvh = h / group;
        const int  p0  = c * ATTN_CHUNK;
        const int  p1  = std::min(len, p0 + ATTN_CHUNK);
        const float *    qh = q + (size_t) h * hd;
        const uint16_t * kb = kc + (size_t) kvh * cap * hd;
        const uint16_t * vb = vc + (size_t) kvh * cap * hd;
        float  sc[ATTN_CHUNK];
        float  m = -std::numeric_limits<float>::infinity();
        for (int j = p0; j < p1; j++) {
            float sj   = scale * dot_f16_f32(kb + (size_t) j * hd, qh, hd);
            sc[j - p0] = sj;
            m          = std::max(m, sj);
        }
        float * dst = part + (size_t) t * stride;
        float * o   = dst + 2;
        std::fill(o, o + hd, 0.0f);
        float l = 0.0f;
        for (int j = p0; j < p1; j++) {
            const float p = std::exp(sc[j - p0] - m);
            l += p;
            axpy_f16(o, p, vb + (size_t) j * hd, hd);
        }
        dst[0] = m;
        dst[1] = l;
    });
    for (int hh = 0; hh < nh; hh++) {
        const float * ph = part + (size_t) hh * n_chunks * stride;
        float M = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < n_chunks; c++) {
            if (ph[(size_t) c * stride + 1] > 0.0f) {
                M = std::max(M, ph[(size_t) c * stride]);
            }
        }
        float * oh = out + (size_t) (h0 + hh) * hd;
        float   acc[MAX_HD];
        std::fill(acc, acc + hd, 0.0f);
        float L = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            const float * pc = ph + (size_t) c * stride;
            if (!(pc[1] > 0.0f)) {
                continue;
            }
            const float e = std::exp(pc[0] - M);
            L += e * pc[1];
            for (int d = 0; d < hd; d++) {
                acc[d] += e * pc[2 + d];
            }
        }
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        for (int d = 0; d < hd; d++) {
            oh[d] = acc[d] * inv;
        }
    }
}

void attention_ref(const float * q, const uint16_t * kc, const uint16_t * vc, int hd, int len,
                   float scale, float * out) {
    std::vector<double> s((size_t) len);
    double m = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < len; j++) {
        double d = 0;
        for (int i = 0; i < hd; i++) {
            d += (double) q[i] * f16_to_f32(kc[(size_t) j * hd + i]);
        }
        s[(size_t) j] = d * scale;
        m             = std::max(m, s[(size_t) j]);
    }
    double l = 0;
    std::vector<double> o((size_t) hd, 0.0);
    for (int j = 0; j < len; j++) {
        double p = std::exp(s[(size_t) j] - m);
        l += p;
        for (int i = 0; i < hd; i++) {
            o[(size_t) i] += p * f16_to_f32(vc[(size_t) j * hd + i]);
        }
    }
    for (int i = 0; i < hd; i++) {
        out[i] = (float) (o[(size_t) i] / l);
    }
}

} // namespace argus_doppel
