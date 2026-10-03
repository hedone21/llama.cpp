// Host tests of the pure logic in argus-dagger.cpp (argus-engine tickets/031 criterion 1).
// Build and run: bash argus-dagger-test.sh

#include "argus-dagger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

using namespace argus_dagger;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                                 \
        }                                                                             \
    } while (0)
#define NEAR(a, b, eps) CHECK(std::fabs((a) - (b)) < (eps))

static const double EPS = 1e-9;

namespace argus_dagger {
struct SchedPeek {
    static const std::optional<Sched::Frame> & frame(const Sched & s) { return s.frame; }
    static double ovh(const Sched & s) { return *s.ovh.value(); }
    static const std::vector<size_t> & chunk_ops(const Sched & s) { return s.chunk_ops; }
};
} // namespace argus_dagger

// ops: per-layer labels for n_layers layers, then head labels.
static std::vector<std::string> ops(int n_layers, const std::vector<std::string> & per, const std::vector<std::string> & head) {
    std::vector<std::string> v;
    for (int l = 0; l < n_layers; ++l) {
        for (const auto & t : per) {
            v.push_back(t + "-" + std::to_string(l));
        }
    }
    for (const auto & t : head) {
        v.push_back(t);
    }
    return v;
}

static std::string table(const std::vector<std::string> & o, const std::vector<std::pair<std::string, std::vector<double>>> & freqs) {
    std::string s = "# test\nversion 1\nops " + std::to_string(o.size()) + "\n";
    for (size_t i = 0; i < o.size(); ++i) {
        s += "op " + std::to_string(i) + " " + o[i] + "\n";
    }
    for (const auto & [k, ms] : freqs) {
        s += "freq " + k + " 5";
        for (double x : ms) {
            char b[32];
            snprintf(b, sizeof(b), " %g", x);
            s += b;
        }
        s += "\n";
    }
    return s;
}

static bool throws_with(const std::string & text, const std::string & needle) {
    try {
        parse_table(text);
    } catch (const std::runtime_error & e) {
        if (std::string(e.what()).find(needle) == std::string::npos) {
            fprintf(stderr, "  (message was: %s)\n", e.what());
            return false;
        }
        return true;
    }
    return false;
}

#define TEST(name) static void name()
#define RUN(name)                                  \
    do {                                           \
        const int before = g_fail;                 \
        name();                                    \
        if (g_fail == before) {                    \
            g_pass++;                              \
            fprintf(stderr, "ok   %s\n", #name);   \
        } else {                                   \
            fprintf(stderr, "FAIL %s\n", #name);   \
        }                                          \
    } while (0)

// (a) Budget against a hand computation.
TEST(budget_matches_hand_computation) {
    NEAR(budget_ms(16.6, 0.0, 4.0), 12.1, EPS);
    NEAR(budget_ms(11.2, 3.3, 1.25), 11.2 - 3.3 - 1.25 - 0.5, EPS);
    NEAR(budget_ms(16.9, 14.0, 4.0), -1.6, EPS);
}

// (b) Greedy: exact fit, one over, and a first op larger than the budget.
TEST(greedy_picks_the_longest_fitting_run) {
    const std::vector<double> lat = { 2.0, 3.0, 1.5, 4.0, 0.5 };
    auto g = [&](double budget) {
        return greedy_ops([&](size_t k) { return k < lat.size() ? lat[k] : 1e9; }, budget);
    };
    auto eq = [](Greedy a, size_t n, bool o) { return a.n == n && a.oversize == o; };
    CHECK(eq(g(6.5), 3, false));   // 2 + 3 + 1.5 = 6.5 exactly
    CHECK(eq(g(10.4), 3, false));  // 6.5 + 4 > 10.4
    CHECK(eq(g(10.9), 4, false));  // 10.5 fits, 11 does not
    CHECK(eq(g(11.0), 5, false));
    CHECK(eq(g(1.9), 1, true));    // first op over the budget → that one op, flagged
    CHECK(eq(g(0.0), 1, true));
    CHECK(eq(g(-3.0), 1, true));
    const std::vector<double> l2 = { 1.0, 5.0, 0.1 };  // consecutive only
    CHECK(eq(greedy_ops([&](size_t k) { return k < l2.size() ? l2[k] : 1e9; }, 3.0), 1, false));
}

// (c) Unprofiled frequency → nearest valid; a tie goes to the lower frequency.
TEST(predicts_from_the_nearest_frequency) {
    const auto o = ops(1, { "A" }, { "H" });
    Predictor p(parse_table(table(o, { { "400", { 4.0, 40.0 } }, { "600", { 6.0, 60.0 } } })));
    CHECK(p.predict(400, 0) == 4.0);
    CHECK(p.predict(600, 1) == 60.0);
    CHECK(p.nearest_valid(450) == 400);
    CHECK(p.predict(450, 0) == 4.0);
    CHECK(p.predict(560, 0) == 6.0);
    CHECK(p.predict(9999, 1) == 60.0);
    CHECK(p.predict(1, 1) == 40.0);
    CHECK(p.nearest_valid(500) == 400);  // 100 from both: the lower one
    CHECK(p.predict(500, 0) == 4.0);
}

// (c') A new frequency is used once every op has NEW_FREQ_MIN_SAMPLES samples.
TEST(new_frequency_needs_samples_for_every_op) {
    const auto o = ops(1, { "A" }, { "H" });
    Predictor p(parse_table(table(o, { { "400", { 4.0, 40.0 } } })));
    for (size_t k = 0; k < NEW_FREQ_MIN_SAMPLES; ++k) {
        p.observe(500, 0, 2.0);
    }
    p.observe(500, 1, 20.0);
    CHECK(!p.is_valid(500));
    CHECK(p.predict(500, 0) == 4.0);
    for (size_t k = 1; k < NEW_FREQ_MIN_SAMPLES; ++k) {
        p.observe(500, 1, 20.0);
    }
    CHECK(p.is_valid(500));
    CHECK(p.predict(500, 0) == 2.0);
    CHECK(p.predict(500, 1) == 20.0);
    CHECK(p.predict(520, 1) == 20.0);
}

// (d) WMA: linear weights 1..n over the last WMA_N samples; a table entry starts from its value.
TEST(wma_matches_its_formula) {
    Wma w = Wma::seeded(4.0);
    CHECK(*w.value() == 4.0);
    w.push(1.0);
    NEAR(*w.value(), 2.0, EPS);  // (1·4 + 2·1) / 3
    w.push(7.0);
    NEAR(*w.value(), 27.0 / 6.0, EPS);
    Wma v;
    CHECK(!v.value());
    for (int x = 1; x <= 10; ++x) {
        v.push(x);
    }
    double num = 0.0;  // window keeps 3..=10: Σ k·(k+2) / Σ k, k = 1..8
    for (int k = 1; k <= 8; ++k) {
        num += k * (k + 2);
    }
    CHECK(v.len() == WMA_N);
    NEAR(*v.value(), num / 36.0, EPS);

    const auto o = ops(1, { "A" }, { "H" });
    Predictor p(parse_table(table(o, { { "400", { 4.0, 40.0 } } })));
    p.observe(400, 0, 1.0);
    NEAR(p.predict(400, 0), 2.0, EPS);
    CHECK(p.predict(400, 1) == 40.0);
}

// (e) Ping: 0.5 ms is idle, 0.51 ms is render with T_render = d; boundary alignment.
TEST(ping_threshold) {
    CHECK(!ping_render(0.0));
    CHECK(!ping_render(0.5));
    CHECK(ping_render(0.51) && *ping_render(0.51) == 0.51);
    CHECK(ping_render(7.25) && *ping_render(7.25) == 7.25);
    NEAR(next_boundary(16.6, 100.0, 107.0, std::make_pair(7.0, 7.0)), 116.9, EPS);
    NEAR(next_boundary(16.6, 100.0, 105.0, std::make_pair(5.0, 7.0)), 114.9, EPS);
    NEAR(next_boundary(16.6, 100.0, 100.2, std::nullopt), 116.8, EPS);
}

// (f) Table: op count mismatch, missing op time, empty frequency table, bad values.
TEST(table_rejects_bad_files) {
    const auto good = ops(28, { "RMS_NORM+MUL:norm", "MUL_MAT:Qcur", "FLASH_ATTN_EXT:kqv" }, { "MUL_MAT:result_output" });
    const std::vector<double> ms(good.size(), 0.1);
    bool ok = true;
    try {
        Table t = parse_table(table(good, { { "900", ms } }));
        ok = t.ops.size() == good.size() && t.freqs.size() == 1;
    } catch (...) {
        ok = false;
    }
    CHECK(ok);
    // `ops` says more than the op lines.
    std::string t1 = table(good, { { "900", ms } });
    const std::string n_line = "ops " + std::to_string(good.size());
    t1.replace(t1.find(n_line), n_line.size(), "ops 999");
    CHECK(throws_with(t1, "op lines"));
    // An op time missing for one frequency.
    CHECK(throws_with(table(good, { { "900", std::vector<double>(good.size() - 1, 0.1) } }), "op times"));
    // No frequencies.
    CHECK(throws_with(table(good, {}), "no frequencies"));
    // A frequency with no times.
    CHECK(throws_with(table(good, { { "900", {} } }), "op times"));
    // A zero time.
    std::vector<double> z = ms;
    z[3] = 0.0;
    CHECK(throws_with(table(good, { { "900", z } }), "positive"));
    // A key that is not MHz.
    CHECK(throws_with(table(good, { { "fast", ms } }), "MHz"));
    // Version.
    std::string t2 = table(good, { { "900", ms } });
    t2.replace(t2.find("version 1"), 9, "version 2");
    CHECK(throws_with(t2, "version"));
    // Round trip through table_text.
    Table t = parse_table(table(good, { { "900", ms } }));
    Table r = parse_table(table_text(t, { "x" }));
    CHECK(r.ops == t.ops && r.freqs == t.freqs && r.samples == t.samples);
}

// A fake GPU: each op takes its scripted time; the clock advances only by sleeps, pings and drains.
struct Fake : Device {
    double t = 0.0;
    std::deque<double> pings;
    uint32_t mhz = 500;
    std::vector<double> op_ms = { 3.0, 4.0, 2.0 };
    std::vector<double> recorded;
    double drain_overhead = 0.0;
    std::vector<std::pair<double, double>> slept;
    std::vector<double> ping_at;

    double now_ms() override { return t; }
    void sleep_until_ms(double t_ms) override {
        slept.push_back({ t, t_ms });
        t = t_ms;
    }
    Ping ping() override {
        double d = 0.1;
        if (!pings.empty()) {
            d = pings.front();
            pings.pop_front();
        }
        ping_at.push_back(t);
        t += d + 0.05;
        return { d, d + 0.05 };
    }
    uint32_t freq_mhz() override { return mhz; }
    std::vector<double> drain() override {
        std::vector<double> v;
        v.swap(recorded);
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        t += s + drain_overhead;
        return v;
    }
    void run_op(Sched & s, const std::string & label, size_t op) {
        s.before_op(*this, label);
        recorded.push_back(op_ms[op]);
    }
};

static const std::vector<std::string> SMALL = { "A-0", "B-0", "H" };

// One layer of two ops (A 3 ms, B 4 ms), head H 2 ms: 9 ms per token.
static Sched small_sched(double frame_ms) {
    return Sched(parse_table(table(SMALL, { { "500", { 3.0, 4.0, 2.0 } } })), "t.txt", frame_ms, nullptr);
}

static void run_token(Sched & s, Fake & f, size_t pos) {
    s.begin_token(f, pos);
    f.run_op(s, "A-0", 0);
    f.run_op(s, "B-0", 1);
    f.run_op(s, "H", 2);
}

// The frame loop: budget 16.6 − 0 − 4 − 0.5 = 12.1 fits A+B+H+A (12) → 4 ops, and the next frame
// starts one period after the first ping's end.
TEST(frame_loop_chunks_across_tokens) {
    Sched s = small_sched(16.6);
    Fake f;
    run_token(s, f, 10);
    run_token(s, f, 11);
    CHECK(f.ping_at.size() == 2);
    CHECK(f.ping_at[0] == 0.0);
    NEAR(f.ping_at[1], 16.75, EPS);
    CHECK(s.frames_total() == 1);
    CHECK(s.oversize_total() == 0);
    NEAR(SchedPeek::ovh(s), (4.0 + 2.0 * 0.15) / 3.0, EPS);
    s.finish(&f);
    CHECK(s.frames_total() == 2);
}

// A render ping sets T_render = d, shrinks the budget, and anchors the boundary on the ping.
TEST(render_ping_shrinks_the_budget) {
    Sched s = small_sched(16.6);
    Fake f;
    f.pings.push_back(5.0);
    s.begin_token(f, 10);
    f.run_op(s, "A-0", 0);
    const auto fr = *SchedPeek::frame(s);
    CHECK(fr.t_render == 5.0);
    CHECK(fr.r_hat == 5.0);
    NEAR(fr.budget, 7.4, EPS);
    CHECK(fr.n_ops == 2);
    NEAR(fr.next_boundary, 16.9, EPS);
    f.run_op(s, "B-0", 1);
    f.run_op(s, "H", 2);
    NEAR(f.slept[0].first, 12.05, EPS);
    NEAR(f.slept[0].second, 16.9, EPS);
    NEAR(f.ping_at[1], 16.9, EPS);
    CHECK(SchedPeek::chunk_ops(s) == std::vector<size_t>{ 2 });
    NEAR(SchedPeek::ovh(s), (4.0 + 2.0 * 0.05) / 3.0, 1e-6);
}

// (g1) A render that started before the ping (d < R̂) pulls the next boundary back by R̂ − d.
TEST(late_render_pulls_the_boundary_back) {
    Sched s = small_sched(16.6);
    Fake f;
    f.pings.push_back(7.0);
    f.pings.push_back(5.0);
    s.begin_token(f, 10);
    f.run_op(s, "A-0", 0);
    NEAR(SchedPeek::frame(s)->next_boundary, 16.9, EPS);
    f.run_op(s, "B-0", 1);
    const auto fr = *SchedPeek::frame(s);
    NEAR(fr.t_issue, 16.9, EPS);
    CHECK(fr.t_render == 5.0 && fr.r_hat == 7.0 && fr.pings == 1);
    NEAR(fr.next_boundary, 31.8, EPS);
    const double ovh = (4.0 + 2.0 * 0.05) / 3.0;
    NEAR(fr.t_ovh, ovh, 1e-9);
    NEAR(fr.budget, 14.9 - 5.0 - ovh - 0.5, 1e-9);
    CHECK(fr.n_ops == 2);
}

// (g2) After a detection, an idle ping is followed by extra pings 0.3 ms apart; the first one that
// detects is the frame's ping.
TEST(extra_pings_find_the_render_start) {
    Sched s = small_sched(16.6);
    Fake f;
    for (double d : { 7.0, 0.1, 0.1, 6.8 }) {
        f.pings.push_back(d);
    }
    s.begin_token(f, 10);
    f.run_op(s, "A-0", 0);
    f.run_op(s, "B-0", 1);
    CHECK(f.ping_at.size() == 4);
    NEAR(f.ping_at[2], 17.35, 1e-9);
    NEAR(f.ping_at[3], 17.8, 1e-9);
    const auto fr = *SchedPeek::frame(s);
    CHECK(fr.pings == 3);
    NEAR(fr.t_issue, 17.8, 1e-9);
    NEAR(fr.probe_ms, 0.9, 1e-9);
    CHECK(fr.t_render == 6.8 && fr.r_hat == 7.0);
    NEAR(fr.next_boundary, 17.8 + 6.8 - 7.0 + 16.9, 1e-9);
}

// (g3) Extra pings stop at the window, and only follow a detection in the last 8 frames.
TEST(extra_pings_stop_at_the_window_and_after_eight_idle_frames) {
    Sched s(parse_table(table(SMALL, { { "500", { 20.0, 20.0, 20.0 } } })), "t.txt", 16.6, nullptr);
    Fake f;
    f.op_ms = { 20.0, 20.0, 20.0 };
    f.pings.push_back(7.0);
    for (size_t pos = 10; pos < 14; ++pos) {
        run_token(s, f, pos);
    }
    s.finish(&f);
    CHECK(s.frames_total() == 12);
    CHECK(f.ping_at.size() == 1 + 8 * 6 + 3);
}

// (g4) † 15: after an oversize frame the next boundary keeps the previous frame's phase
// (previous boundary + k·P) instead of re-anchoring on a render that waited behind our op.
TEST(after_oversize_the_boundary_keeps_the_phase) {
    Sched s = small_sched(16.6);
    Fake f;
    f.pings.push_back(14.0);  // frame 1: budget 16.9 − 14 − 4 − 0.5 < 0 → oversize, A alone
    f.pings.push_back(5.0);   // frame 2: a render seen 5 ms before its end
    s.begin_token(f, 10);
    f.run_op(s, "A-0", 0);
    CHECK(SchedPeek::frame(s)->oversize);
    NEAR(SchedPeek::frame(s)->next_boundary, 16.9, EPS);
    f.run_op(s, "B-0", 1);
    const auto fr = *SchedPeek::frame(s);
    // Frame 2 starts at 14.05 + 3 = 17.05 (past 16.9, no sleep). Re-anchoring would give
    // 17.05 + 5 − 14 + 16.6 + 0.3 = 24.95; the held phase is 16.9 + 16.6 = 33.5.
    NEAR(fr.t_issue, 17.05, 1e-9);
    CHECK(fr.hold);
    NEAR(fr.next_boundary, 33.5, 1e-9);
    const double ovh = (4.0 + 2.0 * 0.05) / 3.0;
    NEAR(fr.budget, (33.5 - 17.05) - 5.0 - ovh - 0.5, 1e-9);
    CHECK(fr.n_ops == 3);  // B 4 + H 2 + A 3 = 9 fits 9.58, + B 13 does not
    // k > 1 when the oversize op ran past more than one period.
    NEAR(held_boundary(16.9, 16.6, 40.0), 16.9 + 2 * 16.6, 1e-9);
    NEAR(held_boundary(16.9, 16.6, 33.5), 16.9 + 2 * 16.6, 1e-9);
}

// A frame that was not oversize re-anchors as before (no hold).
TEST(no_hold_after_a_frame_that_fit) {
    Sched s = small_sched(16.6);
    Fake f;
    f.pings.push_back(7.0);
    f.pings.push_back(5.0);
    s.begin_token(f, 10);
    f.run_op(s, "A-0", 0);
    CHECK(!SchedPeek::frame(s)->oversize);
    f.run_op(s, "B-0", 1);
    CHECK(!SchedPeek::frame(s)->hold);
    NEAR(SchedPeek::frame(s)->next_boundary, 31.8, EPS);
}

// An op larger than the budget still runs alone (never skipped forever).
TEST(oversize_op_runs_alone) {
    Sched s = small_sched(16.6);
    Fake f;
    for (int k = 0; k < 3; ++k) {
        f.pings.push_back(14.0);
    }
    run_token(s, f, 10);
    CHECK(s.oversize_total() == 3);
    CHECK(f.ping_at.size() == 3);
    s.finish(&f);
    CHECK(s.frames_total() == 3);
}

// A dispatched op that differs from the table stops the run.
TEST(rejects_an_op_the_table_does_not_have) {
    Sched s = small_sched(16.6);
    Fake f;
    s.begin_token(f, 10);
    s.before_op(f, "A-0");
    f.recorded.push_back(3.0);
    bool threw = false;
    try {
        s.before_op(f, "X");
    } catch (const std::runtime_error & e) {
        threw = std::string(e.what()).find("op 1") != std::string::npos;
    }
    CHECK(threw);
}

// Profile mode: records the op list from the first token, drops tokens whose frequency moved,
// writes medians for frequencies with enough samples.
TEST(profile_writes_a_table_that_loads) {
    const std::string out = "/tmp/argus-dagger-test-profile.txt";
    std::remove(out.c_str());
    {
        Sched s(out);
        Fake f;
        const uint32_t mhz[] = { 500, 500, 500, 500, 700 };
        for (size_t k = 0; k < 5; ++k) {
            f.mhz = mhz[k];
            s.begin_token(f, 100 + k);
            f.op_ms = { 3.0 + (double) k, 4.0, 2.0 };
            f.run_op(s, "A-0", 0);
            f.run_op(s, "B-0", 1);
            f.run_op(s, "H", 2);
        }
        s.finish(&f);
        CHECK(f.ping_at.empty());
    }
    Table t = load_table(out);
    CHECK(t.ops == SMALL);
    CHECK(t.freqs.size() == 1 && t.freqs.count(500) == 1);
    CHECK((t.freqs[500] == std::vector<double>{ 4.0, 4.0, 2.0 }));
    std::remove(out.c_str());
}

TEST(profile_drops_a_token_whose_frequency_moved) {
    ProfileAcc acc;
    acc.add_token(500, 600, 1, { 1.0 });
    acc.add_token(500, 500, 2, { 1.0 });
    CHECK(acc.tokens == 1 && acc.discarded == 1);
}

// (h) Static Chunking: N=3 over 11 ops → drains after 3, 6, 9; a new token restarts the count.
TEST(static_chunk_drains_every_n_ops_and_restarts_each_token) {
    Chunker c(3);
    c.begin_token();
    std::vector<int> at;
    for (int k = 1; k <= 11; ++k) {
        if (c.after_op()) {
            at.push_back(k);
        }
    }
    CHECK((at == std::vector<int>{ 3, 6, 9 }));
    c.begin_token();
    CHECK(!c.after_op());
    CHECK(!c.after_op());
    CHECK(c.after_op());
    CHECK(c.summary().find("tokens=2 ops=14 drains=4 ops_per_drain=3.500") != std::string::npos);
    Chunker one(1);
    one.begin_token();
    CHECK(one.after_op() && one.after_op());
    bool threw = false;
    try {
        Chunker zero(0);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    RUN(budget_matches_hand_computation);
    RUN(greedy_picks_the_longest_fitting_run);
    RUN(predicts_from_the_nearest_frequency);
    RUN(new_frequency_needs_samples_for_every_op);
    RUN(wma_matches_its_formula);
    RUN(ping_threshold);
    RUN(table_rejects_bad_files);
    RUN(frame_loop_chunks_across_tokens);
    RUN(render_ping_shrinks_the_budget);
    RUN(late_render_pulls_the_boundary_back);
    RUN(extra_pings_find_the_render_start);
    RUN(extra_pings_stop_at_the_window_and_after_eight_idle_frames);
    RUN(after_oversize_the_boundary_keeps_the_phase);
    RUN(no_hold_after_a_frame_that_fit);
    RUN(oversize_op_runs_alone);
    RUN(rejects_an_op_the_table_does_not_have);
    RUN(profile_writes_a_table_that_loads);
    RUN(profile_drops_a_token_whose_frequency_moved);
    RUN(static_chunk_drains_every_n_ops_and_restarts_each_token);
    fprintf(stderr, "%d passed, %d failed checks\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
