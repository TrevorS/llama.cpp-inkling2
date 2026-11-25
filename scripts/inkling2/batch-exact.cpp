// compare target logits and intermediate tensors: K tokens decoded one by one vs as one batch
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

struct capture {
    bool on = false;
    std::vector<std::pair<std::string, std::vector<float>>> rows; // in graph order
    std::map<std::string, int> seen;
};

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (ask) {
        return c->on && t->type == GGML_TYPE_F32 && t->name[0] != '\0' && strncmp(t->name, "leaf", 4) != 0 && strchr(t->name, ' ') == nullptr;
    }
    if (!ggml_is_contiguous(t)) {
        return true;
    }
    std::vector<float> d(ggml_nelements(t));
    ggml_backend_tensor_get(t, d.data(), 0, ggml_nbytes(t));
    const std::string key = std::string(t->name) + "#" + std::to_string(c->seen[t->name]++);
    c->rows.emplace_back(key, std::move(d));
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf K [capture=0|1] [prompt]\n", argv[0]);
        return 1;
    }
    const int K = atoi(argv[2]);
    const bool cap = argc > 3 && atoi(argv[3]) != 0;
    const std::string prompt = argc > 4 ? argv[4] :
        "<|im_start|>user\nWrite a Python function that parses an ISO 8601 duration string into seconds.<|im_end|>\n<|im_start|>assistant\nHere is a function that parses the duration and returns the total number of seconds:";

    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(65536);
    const int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    toks.resize(n);
    if (getenv("NTOK") && atoi(getenv("NTOK")) < n) {
        toks.resize(atoi(getenv("NTOK")));
    }
    const int n0 = (int) toks.size() - K;
    fprintf(stderr, "prompt %d tokens, prefill %d, compare %d\n", (int) toks.size(), n0, K);

    capture ca, cb_;
    auto make = [&](capture * c) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 32768;
        cp.swa_full = false; // the SWA cache is a ring as in llama-server
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.cb_eval = cb;
        cp.cb_eval_user_data = c;
        return llama_init_from_model(model, cp);
    };
    llama_context * a = make(&ca);
    llama_context * b = make(&cb_);

    auto run = [&](llama_context * ctx, int beg, int cnt, bool all_logits) {
        llama_batch bt = llama_batch_init(cnt, 0, 1);
        for (int i = 0; i < cnt; ++i) {
            bt.token[i] = toks[beg + i];
            bt.pos[i] = beg + i;
            bt.n_seq_id[i] = 1;
            bt.seq_id[i][0] = 0;
            bt.logits[i] = all_logits || i == cnt - 1;
        }
        bt.n_tokens = cnt;
        if (llama_decode(ctx, bt) != 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
        llama_batch_free(bt);
    };

    for (int p = 0; p < n0; p += 2048) {
        const int c = std::min(2048, n0 - p);
        run(a, p, c, false);
        run(b, p, c, false);
    }

    // single steps
    std::vector<std::vector<float>> la(K);
    std::vector<decltype(ca.rows)> sa(K);
    for (int j = 0; j < K; ++j) {
        ca.on = cap; ca.rows.clear(); ca.seen.clear();
        run(a, n0 + j, 1, true);
        const float * l = llama_get_logits_ith(a, 0);
        la[j].assign(l, l + n_vocab);
        sa[j] = std::move(ca.rows);
    }
    ca.on = false;

    // batch
    cb_.on = cap; cb_.rows.clear(); cb_.seen.clear();
    run(b, n0, K, true);
    cb_.on = false;

    int n_bad = 0;
    for (int j = 0; j < K; ++j) {
        const float * l = llama_get_logits_ith(b, j);
        double md = 0; int nd = 0, am = 0, bm = 0;
        for (int v = 0; v < n_vocab; ++v) {
            if (l[v] != la[j][v]) { nd++; md = fmax(md, fabs(l[v] - la[j][v])); }
            if (la[j][v] > la[j][am]) am = v;
            if (l[v] > l[bm]) bm = v;
        }
        printf("token %d: logits %s (%d differ, max %.3g), argmax %d vs %d\n", j, nd ? "DIFF" : "same", nd, md, am, bm);
        n_bad += nd != 0;
    }

    if (cap) {
        // first tensor in graph order whose token slices differ
        std::map<std::string, size_t> idx;
        for (size_t i = 0; i < cb_.rows.size(); ++i) idx[cb_.rows[i].first] = i;
        int shown = 0;
        for (const auto & [key, d1] : sa[0]) {
            auto it = idx.find(key);
            if (it == idx.end()) { continue; }
            const auto & db = cb_.rows[it->second].second;
            if (db.size() != d1.size()*K) { continue; }
            bool same = true;
            double md = 0;
            for (int j = 0; j < K; ++j) {
                const auto & sj = sa[j];
                const std::vector<float> * dj = nullptr;
                for (const auto & r : sj) if (r.first == key) { dj = &r.second; break; }
                if (!dj) { same = false; break; }
                for (size_t e = 0; e < d1.size(); ++e) {
                    const float x = db[j*d1.size() + e], y = (*dj)[e];
                    if (x != y && !(std::isnan(x) && std::isnan(y))) { same = false; md = fmax(md, fabs(x - y)); }
                }
            }
            if (!same) {
                printf("DIFF %-40s n=%zu max %.3g\n", key.c_str(), d1.size(), md);
                if (++shown >= 25) break;
            }
        }
        printf("captured %zu single / %zu batch tensors\n", sa[0].size(), cb_.rows.size());
    }

    llama_free(a);
    llama_free(b);
    llama_model_free(model);
    return n_bad ? 2 : 0;
}
