// argus-tf-nll — teacher-forced per-token NLL, in the CSV format of ARGUS `argus-eval --ppl-nll-csv`
// (argus-engine engine/src/session/ppl/runner.rs). The first P tokens go in one llama_decode call (prefill,
// all rows scored except the last); every later token goes in its own llama_decode call (decode path,
// n_tokens = 1). Row i scores token i+1 from the logits after token i. Prefill rows i = 0..P-2, decode rows
// i = P..N-2 (row P-1 is not scored, as in ARGUS).
//
// usage: llama-argus-tf-nll -m model.gguf -f text.txt -p P -o out.csv [-n N] [-t threads]

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double nll_of(const float * logits, int n_vocab, llama_token target) {
    float mx = -INFINITY;
    for (int j = 0; j < n_vocab; j++) {
        mx = logits[j] > mx ? logits[j] : mx;
    }
    double sum = 0.0;
    for (int j = 0; j < n_vocab; j++) {
        sum += std::exp((double) (logits[j] - mx));
    }
    return -((double) logits[target] - (std::log(sum) + (double) mx));
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, out_path;
    int n_prefill = 128, n_max = 0, n_threads = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-m")) model_path = argv[i + 1];
        else if (!strcmp(argv[i], "-f")) text_path = argv[i + 1];
        else if (!strcmp(argv[i], "-o")) out_path = argv[i + 1];
        else if (!strcmp(argv[i], "-p")) n_prefill = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-n")) n_max = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) n_threads = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (model_path.empty() || text_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt -p P -o out.csv [-n N] [-t threads]\n", argv[0]);
        return 1;
    }

    FILE * tf = fopen(text_path.c_str(), "rb");
    if (!tf) { fprintf(stderr, "cannot open %s\n", text_path.c_str()); return 1; }
    std::string text;
    char buf[65536];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), tf)) > 0) text.append(buf, r);
    fclose(tf);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize as ARGUS does for Qwen: no BOS (add_special = false), special text stays plain text.
    int n_tok = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, false);
    std::vector<llama_token> toks(n_tok);
    if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n_tok, false, false) != n_tok) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    if (n_max > 0 && n_max < n_tok) toks.resize(n_max);
    const int N = (int) toks.size();
    if (n_prefill < 2 || n_prefill >= N) { fprintf(stderr, "bad -p %d for %d tokens\n", n_prefill, N); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) N;
    cp.n_batch  = (uint32_t) (n_prefill > 512 ? n_prefill : 512);
    cp.n_ubatch = cp.n_batch;
    if (n_threads > 0) { cp.n_threads = n_threads; cp.n_threads_batch = n_threads; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }
    fprintf(stderr, "[TFNLL] tokens=%d prefill=%d n_ctx=%u n_vocab=%d\n", N, n_prefill, llama_n_ctx(ctx), n_vocab);

    FILE * out = fopen(out_path.c_str(), "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(out, "phase,token_idx,token_id,nll\n");

    llama_batch batch = llama_batch_init(n_prefill, 0, 1);
    for (int i = 0; i < n_prefill; i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n_prefill;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
    double total = 0.0;
    int count = 0;
    for (int i = 0; i < n_prefill - 1; i++) {
        double nll = nll_of(llama_get_logits_ith(ctx, i), n_vocab, toks[i + 1]);
        fprintf(out, "prefill,%d,%d,%.6f\n", i, toks[i + 1], nll);
        total += nll;
        count++;
    }

    for (int i = n_prefill; i < N - 1; i++) {
        batch.token[0]     = toks[i];
        batch.pos[0]       = i;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        double nll = nll_of(llama_get_logits_ith(ctx, 0), n_vocab, toks[i + 1]);
        fprintf(out, "decode,%d,%d,%.6f\n", i, toks[i + 1], nll);
        total += nll;
        count++;
        if ((i - n_prefill) % 1024 == 0) {
            fprintf(stderr, "[TFNLL] pos=%d running PPL=%.4f\n", i, std::exp(total / count));
        }
    }
    fclose(out);
    fprintf(stderr, "[TFNLL] done rows=%d PPL=%.6f\n", count, std::exp(total / count));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
