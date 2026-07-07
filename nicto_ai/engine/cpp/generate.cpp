/*
 * NICTO AI - Fast Autoregressive Generation (C++)
 *
 * Key features:
 * - Pre-allocated output buffer (no torch.cat per token)
 * - Fused top-k + softmax + multinomial sampling
 * - Stop token detection
 * - Minimal Python overhead in the generation loop
 */

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace nicto_engine {

struct GenerateResult {
    std::vector<int64_t> tokens;
    int64_t n_generated;
    bool stopped_early;
};

// Fused top-k filter + softmax + sample
// Given logits [vocab_size], produces next token
int64_t fused_sample(
    const float* logits,     // [vocab_size]
    int64_t vocab_size,
    float temperature,
    int64_t top_k
) {
    std::vector<float> filtered(vocab_size);
    std::memcpy(filtered.data(), logits, vocab_size * sizeof(float));

    // Temperature
    if (temperature != 1.0f) {
        for (int64_t i = 0; i < vocab_size; i++) {
            filtered[i] /= temperature;
        }
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        // Find k-th largest value
        std::vector<float> sorted_vals(filtered.begin(), filtered.end());
        int64_t k = std::min(top_k, vocab_size);
        std::nth_element(
            sorted_vals.begin(), sorted_vals.begin() + k, sorted_vals.end(),
            std::greater<float>()
        );
        float threshold = sorted_vals[k];

        // Mask below threshold
        for (int64_t i = 0; i < vocab_size; i++) {
            if (filtered[i] < threshold) {
                filtered[i] = -1e9f;
            }
        }
    }

    // Softmax (numerically stable)
    float max_val = *std::max_element(filtered.begin(), filtered.end());
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < vocab_size; i++) {
        filtered[i] = std::exp(filtered[i] - max_val);
        sum_exp += filtered[i];
    }
    for (int64_t i = 0; i < vocab_size; i++) {
        filtered[i] /= sum_exp;
    }

    // Multinomial sample
    float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float cumsum = 0.0f;
    for (int64_t i = 0; i < vocab_size; i++) {
        cumsum += filtered[i];
        if (cumsum >= r) {
            return i;
        }
    }
    return vocab_size - 1; // fallback
}

// Pre-allocated generation buffer (avoids torch.cat)
struct GenerationBuffer {
    std::vector<int64_t> buffer;
    int64_t capacity;
    int64_t length;

    GenerationBuffer(int64_t cap) : capacity(cap), length(0) {
        buffer.resize(cap, 0);
    }

    void append(int64_t token) {
        if (length < capacity) {
            buffer[length++] = token;
        }
    }

    const int64_t* data() const { return buffer.data(); }
    int64_t size() const { return length; }
};

// Check if token matches any stop sequence
bool is_stop_token(int64_t token) {
    // Token 10 = newline, 0 = null byte
    // We check for double newline pattern externally
    return false; // handled in Python layer
}

} // namespace nicto_engine
