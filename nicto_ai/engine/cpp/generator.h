/*
 * NICTO AI - Fast Autoregressive Generation (C++)
 *
 * Pre-allocated output buffer eliminates torch.cat per token.
 * Fused sampling reduces per-token overhead.
 * Stop token detection built-in.
 *
 * Key optimizations over Python:
 * - No tensor reallocation per token
 * - Direct memory writes to output buffer
 * - Inline stop token detection
 * - Minimal Python<->C++ boundary crossings
 */

#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace nicto {

struct GenerateResult {
    std::vector<int64_t> tokens;
    int64_t n_generated;
    bool stopped_early;
};

// Pre-allocated generation buffer
class GenerationBuffer {
public:
    std::vector<int64_t> data;
    int64_t capacity;
    int64_t length;

    explicit GenerationBuffer(int64_t cap)
        : data(cap, 0), capacity(cap), length(0) {}

    void append(int64_t token) {
        if (length < capacity) {
            data[length++] = token;
        }
    }

    int64_t operator[](int64_t idx) const { return data[idx]; }
    int64_t size() const { return length; }
    const int64_t* ptr() const { return data.data(); }
};

// ============================================================
// Check if last N bytes match a stop pattern
// ============================================================
bool check_double_newline(const GenerationBuffer& buf) {
    if (buf.length < 2) return false;
    return buf.data[buf.length - 1] == 10 && buf.data[buf.length - 2] == 10;
}

// ============================================================
// Generate tokens (standalone, no model call)
// This is the C++ side of the generation loop.
// The model forward pass is still called from Python.
// ============================================================
GenerateResult generate_tokens(
    const int64_t* prompt_tokens,
    int64_t prompt_len,
    int64_t max_new_tokens,
    const std::vector<int64_t>& sampled_tokens,  // pre-sampled from Python
    int64_t newline_token,
    int64_t vocab_size
) {
    GenerationBuffer buf(prompt_len + max_new_tokens);

    // Copy prompt
    for (int64_t i = 0; i < prompt_len; i++) {
        buf.append(prompt_tokens[i]);
    }

    // Append sampled tokens
    int64_t n_gen = 0;
    bool stopped = false;
    for (int64_t i = 0; i < static_cast<int64_t>(sampled_tokens.size()) && n_gen < max_new_tokens; i++) {
        buf.append(sampled_tokens[i]);
        n_gen++;

        // Early stop on double newline
        if (check_double_newline(buf)) {
            stopped = true;
            break;
        }
    }

    return {buf.data, n_gen, stopped};
}

// ============================================================
// Stop token position finder
// ============================================================
int64_t find_stop_in_tokens(
    const int64_t* tokens,
    int64_t n_tokens
) {
    // Check for double newline (token 10 = \n)
    for (int64_t i = 1; i < n_tokens; i++) {
        if (tokens[i] == 10 && tokens[i - 1] == 10) {
            return i - 1;  // position before the first \n
        }
    }
    return -1;
}

} // namespace nicto
