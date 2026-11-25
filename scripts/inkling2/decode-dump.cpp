// decode a prompt in batches of BATCH tokens (default 1) and write every logit row to a file, to compare two builds or settings bitwise
// usage: [BATCH=k] [N_RS=n] [DEPTH=d] [KEEP=k | ROLL=seed] [SERVER=1] decode-dump model.gguf out.bin [n_tokens] [prompt | @file]; then cmp out1.bin out2.bin
// N_RS sets n_rs_seq (rollback state slots, as with speculative decoding)
// DEPTH decodes that many tokens first, in batches of 512, and does not write their logits
// KEEP and ROLL act like rejected drafts: each batch keeps only its first tokens (KEEP of them, or a random count from the seed ROLL),
// the other tokens of the batch are wrong ones and are removed after the decode
// the SWA cache is a ring as in llama-server (1536 cells, wraps at position 1536); SERVER=1 gives the 4 slot unified layout (4608 cells)
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out.bin [n_tokens] [prompt | @file]\n", argv[0]);
        return 1;
    }
    const int n_max = argc > 3 ? atoi(argv[3]) : 64;
    std::string prompt = argc > 4 ? argv[4] :
        "<|im_start|>user\nWrite a Python function that parses an ISO 8601 duration string into seconds, then explain how it handles weeks.<|im_end|>\n<|im_start|>assistant\nHere is a function that parses the duration and returns the total number of seconds. It splits the string at the T into a date part and a time part, then reads each number with its unit letter:";
    if (!prompt.empty() && prompt[0] == '@') {
        FILE * fp = fopen(prompt.c_str() + 1, "rb");
        if (!fp) {
            fprintf(stderr, "cannot read %s\n", prompt.c_str() + 1);
            return 1;
        }
        prompt.clear();
        char buf[65536];
        size_t n_read;
        while ((n_read = fread(buf, 1, sizeof(buf), fp)) > 0) {
            prompt.append(buf, n_read);
        }
        fclose(fp);
    }

    const int depth = getenv("DEPTH") ? atoi(getenv("DEPTH")) : 0;

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    const int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    if (n < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    toks.resize(std::min(n, depth + n_max));
    if ((int) toks.size() <= depth) {
        fprintf(stderr, "the prompt has %d tokens, DEPTH needs more\n", (int) toks.size());
        return 1;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(4096, depth + n_max + 64);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq = getenv("N_RS") ? atoi(getenv("N_RS")) : 0;
    cp.swa_full = false;
    if (getenv("SERVER") && atoi(getenv("SERVER")) > 0) {
        cp.n_seq_max  = 4;
        cp.kv_unified = true;
    }
    llama_context * ctx = llama_init_from_model(model, cp);

    const int nb = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;

    const int n_keep = getenv("KEEP") ? atoi(getenv("KEEP")) : 0;
    const bool roll  = getenv("ROLL") != nullptr;
    unsigned rng     = roll ? (unsigned) atoi(getenv("ROLL")) : 0;

    {
        llama_batch bp = llama_batch_init(512, 0, 1);
        for (int i0 = 0; i0 < depth; i0 += 512) {
            const int n_cur = std::min(512, depth - i0);
            bp.n_tokens = n_cur;
            for (int i = 0; i < n_cur; ++i) {
                bp.token[i]     = toks[i0 + i];
                bp.pos[i]       = i0 + i;
                bp.n_seq_id[i]  = 1;
                bp.seq_id[i][0] = 0;
                bp.logits[i]    = false;
            }
            if (llama_decode(ctx, bp) != 0) {
                fprintf(stderr, "decode failed at %d\n", i0);
                return 1;
            }
        }
        llama_batch_free(bp);
    }

    FILE * f = fopen(argv[2], "wb");
    llama_batch bt = llama_batch_init(nb, 0, 1);
    int n_rm = 0;
    for (int i0 = depth; i0 < (int) toks.size();) {
        const int n_cur = std::min(nb, (int) toks.size() - i0);
        int keep = n_cur;
        if (roll && n_cur > 1) {
            rng  = rng*1664525u + 1013904223u;
            keep = 1 + (int) ((rng >> 16) % (unsigned) n_cur);
        }
        if (n_keep > 0) {
            keep = std::min(n_keep, n_cur);
        }
        bt.n_tokens = n_cur;
        for (int i = 0; i < n_cur; ++i) {
            bt.token[i]     = i < keep ? toks[i0 + i] : (toks[i0 + i] + 7) % n_vocab;
            bt.pos[i]       = i0 + i;
            bt.n_seq_id[i]  = 1;
            bt.seq_id[i][0] = 0;
            bt.logits[i]    = true;
        }
        if (llama_decode(ctx, bt) != 0) {
            fprintf(stderr, "decode failed at %d\n", i0);
            return 1;
        }
        for (int i = 0; i < keep; ++i) {
            fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, f);
        }
        if (keep < n_cur) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, i0 + keep, -1)) {
                fprintf(stderr, "seq_rm failed at %d\n", i0 + keep);
                return 1;
            }
            n_rm++;
        }
        i0 += keep;
    }
    llama_batch_free(bt);
    fclose(f);
    fprintf(stderr, "wrote %d rows of %d logits, %d batches with removed tokens\n", (int) toks.size() - depth, n_vocab, n_rm);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
