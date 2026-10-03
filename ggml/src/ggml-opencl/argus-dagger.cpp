// Pure logic of the ARGUS comparison arms (see argus-dagger.h).

#include "argus-dagger.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace argus_dagger {

double budget_ms(double span_ms, double t_render_ms, double t_overhead_ms) {
    return span_ms - t_render_ms - t_overhead_ms - MARGIN_MS;
}

std::optional<double> ping_render(double d_ms) {
    if (d_ms > PING_THR_MS) {
        return d_ms;
    }
    return std::nullopt;
}

double held_boundary(double prev_boundary_ms, double frame_ms, double t_after_ms) {
    double nb = prev_boundary_ms + frame_ms;
    while (nb <= t_after_ms) {
        nb += frame_ms;
    }
    return nb;
}

double next_boundary(double frame_ms, double t_issue_ms, double t_after_ms,
                     std::optional<std::pair<double, double>> render) {
    if (render) {
        const double d     = render->first;
        const double r_hat = render->second;
        return t_issue_ms + d - r_hat + frame_ms + ALIGN_EPS_MS;
    }
    return t_after_ms + frame_ms;
}

Wma Wma::seeded(double x) {
    Wma w;
    w.push(x);
    return w;
}

void Wma::push(double x) {
    if (buf.size() == WMA_N) {
        buf.pop_front();
    }
    buf.push_back(x);
}

std::optional<double> Wma::value() const {
    if (buf.empty()) {
        return std::nullopt;
    }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < buf.size(); ++i) {
        const double w = (double) (i + 1);
        num += w * buf[i];
        den += w;
    }
    return num / den;
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ── table file ─────────────────────────────────────────────────────────────

std::string sanitize_label(const std::string & s) {
    std::string r = s;
    for (char & c : r) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            c = '_';
        }
    }
    return r.empty() ? std::string("_") : r;
}

[[noreturn]] static void bad(const std::string & msg) {
    throw std::runtime_error("gpusched table: " + msg);
}

static std::vector<std::string> split_ws(const std::string & line) {
    std::vector<std::string> tok;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            i++;
        }
        size_t j = i;
        while (j < line.size() && line[j] != ' ' && line[j] != '\t') {
            j++;
        }
        if (j > i) {
            tok.push_back(line.substr(i, j - i));
        }
        i = j;
    }
    return tok;
}

static bool to_long(const std::string & s, long * out) {
    char * end = nullptr;
    errno = 0;
    const long v = strtol(s.c_str(), &end, 10);
    if (s.empty() || errno != 0 || !end || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

static bool to_double(const std::string & s, double * out) {
    char * end = nullptr;
    errno = 0;
    const double v = strtod(s.c_str(), &end);
    if (s.empty() || errno != 0 || !end || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

Table parse_table(const std::string & text) {
    Table t;
    long version = -1;
    long n_ops   = -1;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string line = text.substr(start, end - start);
        start = end + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> tok = split_ws(line);
        if (tok.empty()) {
            continue;
        }
        const std::string & kind = tok[0];
        if (kind == "version") {
            if (tok.size() != 2 || !to_long(tok[1], &version)) {
                bad("bad version line: " + line);
            }
        } else if (kind == "ops") {
            if (tok.size() != 2 || !to_long(tok[1], &n_ops) || n_ops <= 0) {
                bad("bad op count line: " + line);
            }
        } else if (kind == "op") {
            long i = -1;
            if (tok.size() != 3 || !to_long(tok[1], &i) || i != (long) t.ops.size()) {
                bad("op line out of order: " + line);
            }
            t.ops.push_back(tok[2]);
        } else if (kind == "freq") {
            long mhz = 0, samples = -1;
            if (tok.size() < 3 || !to_long(tok[1], &mhz) || mhz <= 0) {
                bad("frequency is not MHz > 0: " + line.substr(0, 40));
            }
            if (!to_long(tok[2], &samples) || samples <= 0) {
                bad(std::to_string(mhz) + " MHz has 0 samples");
            }
            std::vector<double> ms;
            for (size_t k = 3; k < tok.size(); ++k) {
                double x = 0.0;
                if (!to_double(tok[k], &x) || !(std::isfinite(x) && x > 0.0)) {
                    bad(std::to_string(mhz) + " MHz op " + std::to_string(k - 3) + " time is not a positive number");
                }
                ms.push_back(x);
            }
            t.freqs[(uint32_t) mhz]   = ms;
            t.samples[(uint32_t) mhz] = (size_t) samples;
        } else {
            bad("unknown line: " + line.substr(0, 40));
        }
    }
    if (version != 1) {
        bad("version " + std::to_string(version) + " (want 1)");
    }
    if (n_ops <= 0 || (size_t) n_ops != t.ops.size()) {
        bad("ops " + std::to_string(n_ops) + " but " + std::to_string(t.ops.size()) + " op lines");
    }
    if (t.freqs.empty()) {
        bad("no frequencies");
    }
    for (const auto & [mhz, ms] : t.freqs) {
        if (ms.size() != t.ops.size()) {
            bad(std::to_string(mhz) + " MHz has " + std::to_string(ms.size()) + " op times, want " +
                std::to_string(t.ops.size()));
        }
    }
    return t;
}

Table load_table(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        bad("cannot read " + path);
    }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        text.append(buf, n);
    }
    fclose(f);
    return parse_table(text);
}

std::string table_text(const Table & t, const std::vector<std::string> & provenance) {
    std::string o = "# argus gpusched op-latency table (llama.cpp OpenCL decode ops)\n";
    for (const auto & p : provenance) {
        o += "# " + p + "\n";
    }
    o += "version 1\n";
    o += "ops " + std::to_string(t.ops.size()) + "\n";
    for (size_t i = 0; i < t.ops.size(); ++i) {
        o += "op " + std::to_string(i) + " " + sanitize_label(t.ops[i]) + "\n";
    }
    char buf[64];
    for (const auto & [mhz, ms] : t.freqs) {
        auto it = t.samples.find(mhz);
        o += "freq " + std::to_string(mhz) + " " + std::to_string(it == t.samples.end() ? 0 : it->second);
        for (double x : ms) {
            snprintf(buf, sizeof(buf), " %.6f", x);
            o += buf;
        }
        o += "\n";
    }
    return o;
}

// ── predictor ──────────────────────────────────────────────────────────────

Predictor::Predictor(const Table & t) : n_ops(t.ops.size()) {
    for (const auto & [f, ms] : t.freqs) {
        Profile p;
        p.valid = true;
        for (double x : ms) {
            p.ops.push_back(Wma::seeded(x));
        }
        profiles[f] = std::move(p);
    }
}

bool Predictor::is_valid(uint32_t mhz) const {
    auto it = profiles.find(mhz);
    return it != profiles.end() && it->second.valid;
}

uint32_t Predictor::nearest_valid(uint32_t mhz) const {
    bool     found = false;
    uint32_t best  = 0;
    uint32_t bestd = 0;
    for (const auto & [f, p] : profiles) {  // ascending, so a tie keeps the lower one
        if (!p.valid) {
            continue;
        }
        const uint32_t d = f > mhz ? f - mhz : mhz - f;
        if (!found || d < bestd) {
            found = true;
            best  = f;
            bestd = d;
        }
    }
    if (!found) {
        throw std::runtime_error("gpusched: no valid frequency");
    }
    return best;
}

double Predictor::predict(uint32_t mhz, size_t op) const {
    const uint32_t f = is_valid(mhz) ? mhz : nearest_valid(mhz);
    return *profiles.at(f).ops.at(op).value();
}

void Predictor::observe(uint32_t mhz, size_t op, double ms) {
    auto it = profiles.find(mhz);
    if (it == profiles.end()) {
        Profile p;
        p.valid = false;
        p.ops.assign(n_ops, Wma());
        it = profiles.emplace(mhz, std::move(p)).first;
    }
    Profile & p = it->second;
    p.ops.at(op).push(std::max(ms, MIN_OP_MS));
    if (!p.valid && std::all_of(p.ops.begin(), p.ops.end(),
                                [](const Wma & w) { return w.len() >= NEW_FREQ_MIN_SAMPLES; })) {
        p.valid = true;
    }
}

std::vector<uint32_t> Predictor::freqs() const {
    std::vector<uint32_t> r;
    for (const auto & [f, p] : profiles) {
        r.push_back(f);
    }
    return r;
}

// ── offline profile ────────────────────────────────────────────────────────

void ProfileAcc::add_token(uint32_t mhz_start, uint32_t mhz_end, size_t pos, const std::vector<double> & ms) {
    if (mhz_start != mhz_end) {
        discarded++;
        return;
    }
    auto & per_op = samples[mhz_start];
    if (per_op.empty()) {
        per_op.assign(ms.size(), {});
    }
    for (size_t i = 0; i < ms.size() && i < per_op.size(); ++i) {
        per_op[i].push_back(std::max(ms[i], MIN_OP_MS));
    }
    tokens++;
    if (!pos_first) {
        pos_first = pos;
    }
    pos_last = pos;
}

Table ProfileAcc::to_table(const std::vector<std::string> & ops,
                           std::vector<std::pair<uint32_t, size_t>> * dropped) const {
    Table t;
    t.ops = ops;
    for (const auto & [mhz, per_op] : samples) {
        size_t fewest = per_op.empty() ? 0 : SIZE_MAX;
        for (const auto & s : per_op) {
            fewest = std::min(fewest, s.size());
        }
        if (fewest < NEW_FREQ_MIN_SAMPLES) {
            if (dropped) {
                dropped->push_back({ mhz, fewest });
            }
            continue;
        }
        std::vector<double> ms;
        for (const auto & s : per_op) {
            ms.push_back(median(s));
        }
        t.freqs[mhz]   = ms;
        t.samples[mhz] = fewest;
    }
    return t;
}

// ── scheduler ──────────────────────────────────────────────────────────────

Sched::Sched(Table table, std::string table_path_, double frame_ms_, FILE * trace_)
    : profile_mode(false), pred(Predictor(table)), frame_ms(frame_ms_), table_path(std::move(table_path_)),
      trace(trace_), ops_(table.ops), ops_fixed(true) {
    if (trace) {
        fprintf(trace,
                "unix_ms,tok,op0,ping_ms,ping_wall_ms,render,t_render_ms,t_overhead_ms,"
                "t_budget_ms,freq_mhz,chunk_ops,pred_ms,actual_ms,wall_ms,over_budget,"
                "oversize,slept_ms,pings,probe_ms,r_hat_ms,phase_hold\n");
    }
}

Sched::Sched(std::string out_) : profile_mode(true), out(std::move(out_)), ops_fixed(false) {}

Sched::~Sched() {
    if (trace) {
        fclose(trace);
    }
}

static std::string fmt_g(double x) {
    char b[64];
    snprintf(b, sizeof(b), "%g", x);
    return b;
}

std::string Sched::start_line() const {
    if (profile_mode) {
        return "[GPUSCHED] profile-out=" + out + " new_freq_samples=" + std::to_string(NEW_FREQ_MIN_SAMPLES);
    }
    std::string fr = "[";
    bool first = true;
    for (uint32_t f : pred->freqs()) {
        fr += (first ? "" : ", ") + std::to_string(f);
        first = false;
    }
    fr += "]";
    return "[GPUSCHED] table=" + table_path + " frame_ms=" + fmt_g(frame_ms) + " thr_ms=" + fmt_g(PING_THR_MS) +
           " ovh0_ms=" + fmt_g(OVH0_MS) + " margin_ms=" + fmt_g(MARGIN_MS) + " wma=linear" + std::to_string(WMA_N) +
           " new_freq_samples=" + std::to_string(NEW_FREQ_MIN_SAMPLES) + " ops_per_token=" +
           std::to_string(ops_.size()) + " freqs=" + fr + " rhat_n=" + std::to_string(RHAT_N) +
           " align_eps_ms=" + fmt_g(ALIGN_EPS_MS) + " probe_gap_ms=" + fmt_g(PROBE_GAP_MS) +
           " probe_win_ms=" + fmt_g(PROBE_WIN_MS) + " probe_recent=" + std::to_string(PROBE_RECENT_FRAMES) +
           " after_oversize=phase_hold";
}

double Sched::unix_ms(Device & dev, double t_ms) {
    if (!epoch) {
        const double u = std::chrono::duration<double, std::milli>(
                             std::chrono::system_clock::now().time_since_epoch()).count();
        epoch = std::make_pair(dev.now_ms(), u);
    }
    return epoch->second + (t_ms - epoch->first);
}

void Sched::begin_token(Device & dev, size_t pos_) {
    if (tok > 0) {
        end_token(dev);
    }
    if (tok > 0 && tok % SUMMARY_EVERY == 0) {
        summary();
    }
    tok++;
    pos    = pos_;
    op_idx = 0;
    if (profile_mode) {
        tok_start = std::make_pair(dev.freq_mhz(), pos_);
    }
}

void Sched::end_token(Device & dev) {
    if (op_idx == 0) {
        tok_start.reset();
        return;
    }
    if (!ops_fixed) {
        if (ops_.empty()) {
            throw std::runtime_error("gpusched: the decode token has no ops");
        }
        ops_fixed = true;
    }
    if (op_idx != ops_.size()) {
        throw std::runtime_error("gpusched: token " + std::to_string(tok) + " dispatched " +
                                 std::to_string(op_idx) + " ops, the table has " + std::to_string(ops_.size()));
    }
    if (profile_mode) {
        const std::vector<double> ms = dev.drain();
        if (ms.size() != ops_.size()) {
            throw std::runtime_error("gpusched profile: " + std::to_string(ms.size()) + " op times for " +
                                     std::to_string(ops_.size()) + " ops");
        }
        const uint32_t end = dev.freq_mhz();
        if (tok_start) {
            acc.add_token(tok_start->first, end, tok_start->second, ms);
            tok_start.reset();
        }
    }
}

void Sched::before_op(Device & dev, const std::string & label_in) {
    const size_t      i     = op_idx;
    const std::string label = sanitize_label(label_in);
    if (ops_fixed) {
        if (i >= ops_.size() || ops_[i] != label) {
            throw std::runtime_error("gpusched: token " + std::to_string(tok) + " op " + std::to_string(i) + " is " +
                                     label + ", the table has " + (i < ops_.size() ? ops_[i] : "(none)"));
        }
    } else {
        ops_.push_back(label);
    }
    op_idx++;
    if (profile_mode) {
        return;
    }
    if (left == 0) {
        next_frame(dev, i);
    }
    left--;
    chunk_ops.push_back(i);
}

void Sched::next_frame(Device & dev, size_t i) {
    std::optional<double> boundary;
    bool hold = false;
    if (frame) {
        const Frame f = *frame;
        frame.reset();
        boundary = f.next_boundary;
        hold     = f.oversize;  // † 15
        close_frame(dev, f);
    }
    double slept_ms = 0.0;
    if (boundary) {
        const double now = dev.now_ms();
        if (*boundary > now) {
            dev.sleep_until_ms(*boundary);
            slept_ms = dev.now_ms() - now;
        }
    }
    // The frame's ping: the first one, or the first extra ping that detects a render.
    const double t_first = dev.now_ms();
    const bool   recent  = last_det && frames_total_ - *last_det <= PROBE_RECENT_FRAMES;
    uint32_t pings = 0;
    double   t_issue, t_after;
    Ping     ping;
    for (;;) {
        t_issue = dev.now_ms();
        ping    = dev.ping();
        t_after = dev.now_ms();
        pings++;
        if (ping_render(ping.d_ms) || !recent || t_after - t_first + PROBE_GAP_MS >= PROBE_WIN_MS) {
            break;
        }
        dev.sleep_until_ms(t_after + PROBE_GAP_MS);
    }
    const std::optional<double> render = ping_render(ping.d_ms);
    const double t_render = render.value_or(0.0);
    const double t_ovh    = *ovh.value();
    double nb, budget, r_hat;
    if (render) {
        const double d = *render;
        if (dets.size() == RHAT_N) {
            dets.pop_front();
        }
        dets.push_back(d);
        last_det = frames_total_;
        r_hat = d;
        for (double x : dets) {
            r_hat = std::max(r_hat, x);
        }
        nb     = next_boundary(frame_ms, t_issue, t_after, std::make_pair(d, r_hat));
        budget = budget_ms(nb - t_issue, d, t_ovh);
    } else {
        nb     = next_boundary(frame_ms, t_issue, t_after, std::nullopt);
        budget = budget_ms(frame_ms, 0.0, t_ovh);
        r_hat  = 0.0;
    }
    if (hold) {
        // † 15: the previous frame ran one op past its budget; a render seen now may have waited
        // behind it, so keep the boundary on the previous one's phase.
        nb = held_boundary(*boundary, frame_ms, t_after);
        budget = budget_ms(nb - t_issue, t_render, t_ovh);
        holds_total_++;
    }
    const uint32_t mhz = dev.freq_mhz();
    const size_t   n   = ops_.size();
    const Predictor & p = *pred;
    const Greedy g = greedy_ops([&](size_t k) { return p.predict(mhz, (i + k) % n); }, budget);
    double pred_sum = 0.0;
    for (size_t k = 0; k < g.n; ++k) {
        pred_sum += p.predict(mhz, (i + k) % n);
    }
    if (g.oversize) {
        oversize_total_++;
    }
    left = g.n;
    chunk_ops.clear();
    frame = Frame{ t_issue, ping, t_render, t_ovh, budget, mhz, nb, g.n, pred_sum, g.oversize,
                   slept_ms, tok, i, pings, t_issue - t_first, r_hat, hold };
}

void Sched::close_frame(Device & dev, const Frame & f) {
    const std::vector<double> ms = dev.drain();
    const double t_done = dev.now_ms();
    if (ms.size() != chunk_ops.size()) {
        throw std::runtime_error("gpusched: " + std::to_string(ms.size()) + " op times for a chunk of " +
                                 std::to_string(chunk_ops.size()) + " ops");
    }
    double actual = 0.0;
    for (size_t k = 0; k < ms.size(); ++k) {
        pred->observe(f.mhz, chunk_ops[k], ms[k]);
        actual += ms[k];
    }
    const double wall = t_done - f.t_issue;
    ovh.push(std::max(wall - f.t_render - actual, 0.0));
    const bool   over = actual > f.budget;
    const double util = actual / (frame_ms - f.t_render);
    frames_total_++;

    Window & w = window;
    w.frames++;
    w.pings += f.pings;
    if (f.t_render > 0.0) {
        w.render++;
    }
    w.budget.push_back(f.budget);
    w.ops.push_back((double) chunk_ops.size());
    w.util.push_back(util);
    if (over) {
        w.over++;
    }
    w.mhz[f.mhz]++;

    const double t_unix = unix_ms(dev, f.t_issue);
    if (trace) {
        fprintf(trace, "%.3f,%llu,%zu,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%u,%zu,%.4f,%.4f,%.4f,%d,%d,%.4f,%u,%.4f,%.4f,%d\n",
                t_unix, (unsigned long long) f.tok, f.op0, f.ping.d_ms, f.ping.wall_ms, f.t_render > 0.0 ? 1 : 0,
                f.t_render, f.t_ovh, f.budget, f.mhz, chunk_ops.size(), f.pred_sum, actual, wall, over ? 1 : 0,
                f.oversize ? 1 : 0, f.slept_ms, f.pings, f.probe_ms, f.r_hat, f.hold ? 1 : 0);
    }
}

void Sched::summary() {
    if (profile_mode) {
        return;
    }
    Window w = std::move(window);
    window   = Window();
    if (w.frames == 0) {
        return;
    }
    uint32_t freq_mode = 0;
    size_t   best      = 0;
    for (const auto & [f, c] : w.mhz) {  // ascending: on a tie the lower frequency
        if (c > best) {
            best      = c;
            freq_mode = f;
        }
    }
    fprintf(stderr,
            "[GPUSCHED] tok=%llu pos=%zu frames=%zu render_frac=%.4f budget_med=%.3f chunk_ops_med=%g "
            "util_med=%.4f overrun=%.4f oversize=%llu freq_mode=%u ovh_ms=%.3f prefill_graphs=%llu "
            "pings_per_frame=%.3f phase_holds=%llu\n",
            (unsigned long long) tok, pos, w.frames, (double) w.render / w.frames, median(w.budget),
            median(w.ops), median(w.util), (double) w.over / w.frames, (unsigned long long) oversize_total_,
            freq_mode, ovh.value().value_or(NAN), (unsigned long long) prefill_graphs,
            (double) w.pings / w.frames, (unsigned long long) holds_total_);
    fflush(stderr);
    if (trace) {
        fflush(trace);
    }
}

void Sched::finish(Device * dev) {
    if (dev) {
        if (tok > 0) {
            end_token(*dev);
        }
        if (frame) {
            const Frame f = *frame;
            frame.reset();
            close_frame(*dev, f);
        }
    }
    left = 0;
    summary();
    if (trace) {
        fflush(trace);
    }
    if (profile_mode && acc.tokens > 0) {
        std::vector<std::pair<uint32_t, size_t>> dropped;
        const Table t = acc.to_table(ops_, &dropped);
        std::vector<std::string> prov;
        prov.push_back("writer: llama.cpp ggml-opencl ARGUS_GPUSCHED_PROFILE_OUT (argus-engine tickets/031)");
        prov.push_back("unix_s: " + std::to_string((long long) std::chrono::duration_cast<std::chrono::seconds>(
                                                        std::chrono::system_clock::now().time_since_epoch()).count()));
        prov.push_back("tokens: " + std::to_string(acc.tokens) + " discarded_tokens: " + std::to_string(acc.discarded));
        prov.push_back("pos_first: " + std::to_string(acc.pos_first.value_or(0)) + " pos_last: " +
                       std::to_string(acc.pos_last));
        FILE * f = fopen(out.c_str(), "wb");
        if (!f) {
            throw std::runtime_error("gpusched profile: cannot write " + out);
        }
        const std::string text = table_text(t, prov);
        const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
        if (fclose(f) != 0 || !ok) {
            throw std::runtime_error("gpusched profile: cannot write " + out);
        }
        std::string fr, dr;
        for (const auto & [mhz, s] : t.samples) {
            fr += (fr.empty() ? "" : ",") + std::to_string(mhz) + ":" + std::to_string(s);
        }
        for (const auto & [mhz, s] : dropped) {
            dr += (dr.empty() ? "" : ",") + std::to_string(mhz) + ":" + std::to_string(s);
        }
        fprintf(stderr, "[GPUSCHED] profile written=%s tokens=%llu discarded=%llu freqs=[%s] dropped=[%s]\n",
                out.c_str(), (unsigned long long) acc.tokens, (unsigned long long) acc.discarded, fr.c_str(),
                dr.c_str());
        fflush(stderr);
    }
    tok = 0;
}

// ── Static Chunking ────────────────────────────────────────────────────────

Chunker::Chunker(size_t n_) : n(n_) {
    if (n == 0) {
        throw std::runtime_error("static chunk of 0 ops");
    }
}

void Chunker::begin_token() {
    count = 0;
    tokens++;
}

bool Chunker::after_op() {
    count++;
    ops++;
    const bool drain = count % n == 0;
    if (drain) {
        drains++;
    }
    return drain;
}

std::string Chunker::start_line() const {
    return "[STATICCHUNK] start ops_per_chunk=" + std::to_string(n);
}

std::string Chunker::summary() const {
    char b[256];
    snprintf(b, sizeof(b), "[STATICCHUNK] summary tokens=%llu ops=%llu drains=%llu ops_per_drain=%.3f prefill_graphs=%llu",
             (unsigned long long) tokens, (unsigned long long) ops, (unsigned long long) drains,
             drains > 0 ? (double) ops / (double) drains : 0.0, (unsigned long long) prefill_graphs);
    return b;
}

} // namespace argus_dagger
