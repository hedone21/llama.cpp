// Host tests of the pure logic in argus-doppel.cpp (argus-engine tickets/032 criterion 1).
// Controller tests mirror engine/src/layers/tp_controller.rs (ticket 021) input for input.
// Build and run: bash argus-doppel-test.sh   (on the device: the same file built with the NDK)

#include "argus-doppel.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace argus_doppel;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                                 \
        }                                                                             \
    } while (0)

static const int N_HEADS = 12;
static const int FFN     = 8960;

// Asymmetric observation of a segment with GPU speed v_g and CPU speed v_c (share per ms): the
// GPU time is visible only when the CPU had to wait, or on a serial token.
static Obs sim(const SegState & seg, float v_g, float v_c) {
    float r      = seg.applied_share();
    float t_gpu  = r / v_g;
    float t_cpu  = (1.0f - r) / v_c;
    bool  waited = t_gpu > t_cpu;
    Obs   o{waited, t_cpu, std::nullopt};
    if (waited || seg.serial()) {
        o.t_gpu = t_gpu;
    }
    return o;
}

static float r_star(float v_g, float v_c) { return v_g / (v_g + v_c); }

static float quanta_off(const SegState & seg, float r) {
    return std::fabs(seg.applied_share() - r) * (float) seg.geom.total / (float) seg.geom.unit();
}

static const float PAIRS[6][2] = {{1, 4}, {1, 2}, {2, 3}, {3, 2}, {2, 1}, {4, 1}};

static void controller_converges() {
    TpConfig cfg;
    for (SegGeom geom : {SegGeom::attn(N_HEADS), SegGeom::ffn(FFN)}) {
        for (const auto & pr : PAIRS) {
            float v_g = pr[0], v_c = pr[1];
            for (float r0 : {0.25f, 0.75f}) {
                float              target = r_star(v_g, v_c);
                SegState           seg(geom, r0);
                std::vector<float> shares{seg.applied_share()};
                int                steps = 0;
                bool               ok    = true;
                while (seg.phase != Phase::Lookup) {
                    seg.observe(sim(seg, v_g, v_c), cfg);
                    shares.push_back(seg.applied_share());
                    if (++steps > 10) {
                        ok = false;
                        break;
                    }
                }
                CHECK(ok);
                CHECK(quanta_off(seg, target) <= 1.0f + 1e-3f);
                // Sign check: a slower GPU never gets more work on the way down from 0.75.
                if (v_g < v_c && r0 == 0.75f) {
                    auto band = [&](float s) {
                        return std::fabs(s - target) * (float) geom.total / (float) geom.unit() <= 1.0f;
                    };
                    size_t first = 0;
                    while (first < shares.size() && !band(shares[first])) {
                        first++;
                    }
                    CHECK(first < shares.size());
                    for (size_t i = 1; i <= first && i < shares.size(); i++) {
                        CHECK(shares[i] <= shares[i - 1]);
                    }
                }
            }
        }
    }
}

static SegState settled(SegGeom geom, float v_g, float v_c, const TpConfig & cfg) {
    SegState seg(geom, 0.5f);
    for (int i = 0; i < 20; i++) {
        if (seg.phase == Phase::Lookup) {
            return seg;
        }
        seg.observe(sim(seg, v_g, v_c), cfg);
    }
    fprintf(stderr, "  settled(): did not settle\n");
    g_fail++;
    return seg;
}

static Obs obs2(float t_cpu, float t_gpu) {
    Obs o{t_gpu > t_cpu, t_cpu, std::nullopt};
    if (t_gpu > t_cpu) {
        o.t_gpu = t_gpu;
    }
    return o;
}

static TpController lookup_ctl(const TpConfig & cfg) {
    TpController c(2, N_HEADS, FFN, 0.5f, 8, true, cfg);
    for (int layer = 0; layer < 2; layer++) {
        SegState & s = c.seg(layer, 1);
        s.phase      = Phase::Lookup;
        s.q_opt      = s.applied();
        s.t_best     = 10.0f;
        s.t_cpu_best = 9.0f;
    }
    return c;
}

static void controller_contention_paths() {
    TpConfig cfg;
    cfg.probe = false;
    {   // t_gpu alone 1.3x -> Descent, threads unchanged.
        TpController c = lookup_ctl(cfg);
        for (int i = 0; i < 3; i++) {
            c.observe(0, 1, obs2(9.0f, 13.0f));
            CHECK(!c.end_token());
        }
        CHECK(c.seg(0, 1).phase == Phase::Descent);
        CHECK(c.threads == 8);
        CHECK(c.contention == 1);
    }
    {   // t_cpu alone 1.3x -> one step down, every segment back to Startup.
        TpController c = lookup_ctl(cfg);
        for (int t = 0; t < 3; t++) {
            c.observe(0, 1, obs2(13.0f, 10.0f));
            std::optional<int> changed = c.end_token();
            if (t == 2) {
                CHECK(changed && *changed == 6);
            } else {
                CHECK(!changed);
            }
        }
        CHECK(c.threads == 6);
        for (const SegState & s : c.segs) {
            CHECK(s.phase == Phase::Startup);
        }
        CHECK(c.serial(1, 0));
    }
    {   // 1.1x is inside the band -> nothing.
        TpController c = lookup_ctl(cfg);
        for (int i = 0; i < 10; i++) {
            c.observe(0, 1, obs2(9.0f, 11.0f));
            c.end_token();
        }
        CHECK(c.seg(0, 1).phase == Phase::Lookup);
        CHECK(c.contention == 0);
    }
    {   // Fewer than three consecutive slow tokens -> nothing.
        TpController c = lookup_ctl(cfg);
        for (int i = 0; i < 12; i++) {
            bool slow = i % 3 != 2;
            c.observe(0, 1, slow ? obs2(9.0f, 13.0f) : obs2(9.0f, 10.0f));
            c.end_token();
        }
        CHECK(c.seg(0, 1).phase == Phase::Lookup);
        CHECK(c.contention == 0);
        CHECK(c.threads == 8);
    }
}

static void controller_release_probe() {
    const float v_g = 1.0f, v_c = 2.0f;
    SegGeom     geom = SegGeom::ffn(FFN);
    {   // Without the probe: after the GPU doubles its speed, the share never moves.
        TpConfig no_probe;
        no_probe.probe = false;
        SegState seg   = settled(geom, v_g, v_c, no_probe);
        int      held  = seg.applied();
        for (int i = 0; i < 64; i++) {
            seg.observe(sim(seg, 2.0f * v_g, v_c), no_probe);
            CHECK(seg.applied() == held);
            CHECK(seg.phase == Phase::Lookup);
        }
    }
    {   // With the probe: Descent within 9 tokens, then the new r* within 10 steps.
        TpConfig cfg;
        SegState seg    = settled(geom, v_g, v_c, cfg);
        int      tokens = 0;
        while (seg.phase == Phase::Lookup && tokens <= 9) {
            seg.observe(sim(seg, 2.0f * v_g, v_c), cfg);
            tokens++;
        }
        CHECK(tokens <= 9);
        CHECK(seg.phase == Phase::Descent);
        float target = r_star(2.0f * v_g, v_c);
        int   steps  = 0;
        while (seg.phase != Phase::Lookup && steps <= 10) {
            seg.observe(sim(seg, 2.0f * v_g, v_c), cfg);
            steps++;
        }
        CHECK(steps <= 10);
        CHECK(quanta_off(seg, target) <= 1.0f + 1e-3f);
    }
    {   // Speeds unchanged: every probe comes back to the held share, no Descent.
        TpConfig cfg;
        SegState seg  = settled(geom, v_g, v_c, cfg);
        int      held = seg.q_opt;
        for (int i = 0; i < 64; i++) {
            seg.observe(sim(seg, v_g, v_c), cfg);
            CHECK(seg.phase == Phase::Lookup);
            CHECK(seg.applied() == held || seg.applied() == held + 1);
        }
    }
}

// The S25 trace pattern (ticket 021 criterion 6): at the boundary quantum the CPU waits on most
// tokens but not all. Descent must still settle without the cap, and a lone unwaited probe token
// must not restart Descent.
static void controller_noisy_boundary() {
    TpConfig cfg;
    SegGeom  geom  = SegGeom::ffn(FFN);
    const int q_b  = 51;
    unsigned flips = 0;
    auto obs_at = [&](const SegState & seg) {
        int   q     = seg.applied();
        float t_cpu = 1.35f - 0.02f * ((float) q - 50.0f);
        bool  waited;
        if (q == q_b) {
            flips++;
            waited = flips % 3 != 0;
        } else {
            waited = q > q_b;
        }
        float t_gpu = t_cpu + (waited ? 0.1f : -0.1f);
        Obs   o{waited, t_cpu, std::nullopt};
        if (waited || seg.serial()) {
            o.t_gpu = t_gpu;
        }
        return o;
    };
    for (float r0 : {0.75f, 0.5f}) {
        SegState seg(geom, r0);
        int      steps  = 0;
        bool     forced = false;
        while (seg.phase != Phase::Lookup && steps < 100) {
            if (seg.observe(obs_at(seg), cfg) == SegEvent::ConvergedForced) {
                forced = true;
            }
            steps++;
        }
        CHECK(!forced);
        CHECK(steps <= 12);
        CHECK(std::abs(seg.q_opt - q_b) <= 1);
        int hits = 0;
        for (int i = 0; i < 1024; i++) {
            if (seg.observe(obs_at(seg), cfg) == SegEvent::ProbeHit) {
                hits++;
            }
        }
        CHECK(hits == 0);
    }
}

// Not in the Rust tests (they pass with the gradient sign reversed: the bracket hides a one-quantum
// step). A GPU three times slower than the CPU in Descent must move the share down by the gradient
// (0.25 * (1 - 3) / 4 = -0.125, about 9 FFN quanta), not just one quantum.
static void controller_descent_gradient_sign() {
    TpConfig cfg;
    SegState seg(SegGeom::ffn(FFN), 0.5f);
    seg.observe(Obs{true, 1.0f, 1.0f}, cfg); // Startup -> Descent at the balance point
    CHECK(seg.phase == Phase::Descent);
    while (seg.applied() < 60) {
        seg.observe(Obs{false, 1.0f, std::nullopt}, cfg); // GPU slack: probe up
        if (seg.phase != Phase::Descent) {
            break;
        }
    }
    CHECK(seg.phase == Phase::Descent);
    int q_before = seg.applied();
    seg.observe(Obs{true, 1.0f, 3.0f}, cfg);
    CHECK(seg.phase == Phase::Descent);
    CHECK(q_before - seg.applied() >= 8);
}

static void quantization_bounds() {
    for (float r : {0.0f, 1e-6f, 0.04f, 0.5f, 0.96f}) {
        std::optional<int> h_g = quantize_attn(r, N_HEADS);
        CHECK(h_g && *h_g >= 1 && *h_g <= 11);
        std::optional<int> s = quantize_ffn(r, FFN);
        CHECK(s && *s >= 128 && *s <= 8832 && *s % 128 == 0);
    }
    CHECK(!quantize_attn(1.0f, N_HEADS));
    CHECK(!quantize_ffn(1.0f, FFN));
    CHECK(quantize_attn(0.5f, N_HEADS) == 6);
    CHECK(quantize_ffn(0.5f, FFN) == 4480);
}

static void rowslice_offsets_cover_dim() {
    // GPU range [0, split) and CPU range [split, total) cover the axis without overlap.
    for (int h_g = 1; h_g <= 11; h_g++) {
        int gpu_hi = h_g * 128, cpu_lo = h_g * 128, cpu_hi = N_HEADS * 128;
        CHECK(gpu_hi == cpu_lo && cpu_hi == 1536 && gpu_hi > 0 && cpu_lo < cpu_hi);
    }
    SegGeom g = SegGeom::ffn(FFN);
    for (int q = 1; q <= g.q_max(); q++) {
        int s = g.split(q);
        CHECK(s % 128 == 0 && s >= 128 && s <= 8832);
        std::vector<int> cover((size_t) FFN, 0);
        for (int r = 0; r < s; r++) {
            cover[(size_t) r]++;
        }
        for (int r = s; r < FFN; r++) {
            cover[(size_t) r]++;
        }
        bool once = true;
        for (int c : cover) {
            once = once && c == 1;
        }
        CHECK(once);
    }
}

static std::vector<uint16_t> rand_f16(std::mt19937 & rng, size_t n, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<uint16_t>                 v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = f32_to_f16(d(rng));
    }
    return v;
}

static std::vector<float> rand_f32(std::mt19937 & rng, size_t n, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float>                    v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = d(rng);
    }
    return v;
}

static void f16_roundtrip() {
    for (float f : {0.0f, 1.0f, -2.5f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f, 0.333251953125f}) {
        CHECK(f16_to_f32(f32_to_f16(f)) == f);
    }
    CHECK(f32_to_f16(1.0f) == 0x3C00);
    CHECK(f16_to_f32(0xC000) == -2.0f);
}

static void colslice_gemv_matches_dense(Pool & pool) {
    std::mt19937          rng(32);
    const int             rows = 1536, cols = FFN;
    std::vector<uint16_t> w = rand_f16(rng, (size_t) rows * cols, 0.05f);
    std::vector<float>    x = rand_f32(rng, (size_t) cols, 1.0f);
    std::vector<float>    dense((size_t) rows);
    GemvJob               full{w.data(), (size_t) cols, dense.data(), rows};
    gemv_f16_multi(pool, x.data(), cols, &full, 1);
    // Reference in double.
    std::vector<double> ref((size_t) rows, 0.0);
    double              ymax = 0;
    for (int r = 0; r < rows; r++) {
        for (int i = 0; i < cols; i++) {
            ref[(size_t) r] += (double) f16_to_f32(w[(size_t) r * cols + i]) * x[(size_t) i];
        }
        ymax = std::max(ymax, std::fabs(ref[(size_t) r]));
    }
    double err = 0;
    for (int r = 0; r < rows; r++) {
        err = std::max(err, std::fabs(dense[(size_t) r] - ref[(size_t) r]));
    }
    CHECK(err <= 1e-3 * ymax);
    const int pairs[5][2] = {{0, 128}, {128, 4480}, {4480, 8960}, {1024, 8832}, {0, 8960}};
    for (const auto & p : pairs) {
        int                lo = p[0], hi = p[1];
        std::vector<float> a((size_t) rows), b((size_t) rows), c((size_t) rows);
        GemvJob ja{w.data() + lo, (size_t) cols, a.data(), rows};
        gemv_f16_multi(pool, x.data() + lo, hi - lo, &ja, 1);
        GemvJob jb{w.data(), (size_t) cols, b.data(), rows};
        gemv_f16_multi(pool, x.data(), lo, &jb, 1);
        GemvJob jc{w.data() + hi, (size_t) cols, c.data(), rows};
        gemv_f16_multi(pool, x.data() + hi, cols - hi, &jc, 1);
        double e = 0;
        for (int r = 0; r < rows; r++) {
            double y = (double) a[(size_t) r] + b[(size_t) r] + c[(size_t) r];
            e        = std::max(e, std::fabs(y - ref[(size_t) r]));
        }
        CHECK(e <= 1e-3 * ymax);
    }
    // Row slice: rows [s, 8960) of an up-like [8960 x 1536] matrix equal the dense rows.
    std::vector<uint16_t> wu = rand_f16(rng, (size_t) FFN * rows, 0.05f);
    std::vector<float>    xu = rand_f32(rng, (size_t) rows, 1.0f);
    std::vector<float>    yd((size_t) FFN), ys((size_t) FFN);
    GemvJob               jd{wu.data(), (size_t) rows, yd.data(), FFN};
    gemv_f16_multi(pool, xu.data(), rows, &jd, 1);
    const int s = 6656;
    GemvJob   js{wu.data() + (size_t) s * rows, (size_t) rows, ys.data(), FFN - s};
    gemv_f16_multi(pool, xu.data(), rows, &js, 1);
    bool same = true;
    for (int r = 0; r < FFN - s; r++) {
        same = same && ys[(size_t) r] == yd[(size_t) (s + r)];
    }
    CHECK(same);
}

static void attn_group_split_matches_full(Pool & pool) {
    std::mt19937 rng(7);
    const int    hd = 128, n_kv = 2, cap = 600;
    const float  scale = 1.0f / std::sqrt((float) hd);
    for (int len : {1, 37, 512}) {
        std::vector<uint16_t> kc = rand_f16(rng, (size_t) n_kv * cap * hd, 1.0f);
        std::vector<uint16_t> vc = rand_f16(rng, (size_t) n_kv * cap * hd, 1.0f);
        std::vector<float>    q  = rand_f32(rng, (size_t) N_HEADS * hd, 1.0f);
        AttnScratch           sc;
        std::vector<float>    full((size_t) N_HEADS * hd);
        attention_heads(pool, q.data(), kc.data(), vc.data(), cap, N_HEADS, n_kv, hd, 0, N_HEADS, len,
                        scale, full.data(), sc);
        // Reference per head.
        double err_ref = 0;
        for (int h = 0; h < N_HEADS; h++) {
            int                kvh = h / (N_HEADS / n_kv);
            std::vector<float> o((size_t) hd);
            attention_ref(q.data() + (size_t) h * hd, kc.data() + (size_t) kvh * cap * hd,
                          vc.data() + (size_t) kvh * cap * hd, hd, len, scale, o.data());
            for (int d = 0; d < hd; d++) {
                err_ref = std::max(err_ref, (double) std::fabs(o[(size_t) d] - full[(size_t) h * hd + d]));
            }
        }
        CHECK(err_ref <= 1e-5);
        for (int h_g : {1, 5, 6, 7, 11}) {
            std::vector<float> part((size_t) N_HEADS * hd, 0.0f);
            attention_heads(pool, q.data(), kc.data(), vc.data(), cap, N_HEADS, n_kv, hd, 0, h_g, len,
                            scale, part.data(), sc);
            attention_heads(pool, q.data(), kc.data(), vc.data(), cap, N_HEADS, n_kv, hd, h_g, N_HEADS,
                            len, scale, part.data(), sc);
            double err = 0;
            for (size_t i = 0; i < part.size(); i++) {
                err = std::max(err, (double) std::fabs(part[i] - full[i]));
            }
            CHECK(err <= 1e-5);
        }
    }
}

static void rope_neox_matches_reference() {
    std::mt19937 rng(11);
    const int    hd = 128, n_dims = 128;
    const float  base = 1000000.0f;
    for (int pos : {0, 1, 1000, 8191}) {
        std::vector<float> x = rand_f32(rng, (size_t) N_HEADS * hd, 1.0f);
        std::vector<float> y = x;
        rope_neox(y.data(), 3, 9, hd, n_dims, (double) pos, base);
        double err = 0;
        for (int h = 0; h < N_HEADS; h++) {
            for (int i = 0; i < hd / 2; i++) {
                const float * v  = x.data() + (size_t) h * hd;
                std::complex<double> z(v[i], v[i + hd / 2]);
                if (h >= 3 && h < 9) {
                    // Same float angle as the GPU kernel, rotation in double.
                    float th = (float) pos * std::pow(base, (-1.0f / (float) n_dims) * (float) (2 * i));
                    z *= std::polar(1.0, (double) th);
                }
                const float * w = y.data() + (size_t) h * hd;
                err = std::max(err, std::abs(z - std::complex<double>(w[i], w[i + hd / 2])));
            }
        }
        CHECK(err <= 1e-5);
    }
}

static void pool_runs_every_task_once() {
    Pool pool(6);
    for (int round = 0; round < 2000; round++) {
        if (round % 97 == 0) {
            pool.set_active(2 + round % 5);
        }
        int                            n = 1 + round % 53;
        std::vector<std::atomic<int>>  hits((size_t) n);
        for (auto & h : hits) {
            h.store(0);
        }
        pool.run(n, [&](int i) { hits[(size_t) i].fetch_add(1); });
        bool once = true;
        for (auto & h : hits) {
            once = once && h.load() == 1;
        }
        CHECK(once);
    }
}

// The contention knob: after set_active(n) at most n threads run tasks, and set_active() back up
// wakes the workers it had parked.
static void pool_set_active_limits_threads() {
    Pool pool(6);
    auto threads_used = [&](int rounds) {
        std::mutex                  mu;
        std::set<std::thread::id>   ids;
        for (int r = 0; r < rounds; r++) {
            pool.run(24, [&](int) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                std::lock_guard<std::mutex> lk(mu);
                ids.insert(std::this_thread::get_id());
            });
        }
        return ids.size();
    };
    pool.set_active(2);
    CHECK(threads_used(20) <= 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // the parked workers fall asleep
    pool.set_active(6);
    CHECK(threads_used(20) > 2);
}

int main() {
    struct T {
        const char * name;
        void (*fn)();
    };
    static Pool pool4(4);
    const T tests[] = {
        {"quantization_bounds", quantization_bounds},
        {"controller_converges", controller_converges},
        {"controller_contention_paths", controller_contention_paths},
        {"controller_release_probe", controller_release_probe},
        {"controller_noisy_boundary", controller_noisy_boundary},
        {"controller_descent_gradient_sign", controller_descent_gradient_sign},
        {"rowslice_offsets_cover_dim", rowslice_offsets_cover_dim},
        {"f16_roundtrip", f16_roundtrip},
        {"colslice_gemv_matches_dense", [] { colslice_gemv_matches_dense(pool4); }},
        {"attn_group_split_matches_full", [] { attn_group_split_matches_full(pool4); }},
        {"rope_neox_matches_reference", rope_neox_matches_reference},
        {"pool_runs_every_task_once", pool_runs_every_task_once},
        {"pool_set_active_limits_threads", pool_set_active_limits_threads},
    };
    for (const T & t : tests) {
        int before = g_fail;
        t.fn();
        if (g_fail == before) {
            g_pass++;
            printf("ok   %s\n", t.name);
        } else {
            printf("FAIL %s\n", t.name);
        }
    }
    printf("%d passed, %d failed checks\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
