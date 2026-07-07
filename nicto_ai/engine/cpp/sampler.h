/*
 * NICTO AI - Fast Token Sampler (C++)
 *
 * Fused top-k filtering + softmax + multinomial sampling.
 * Single function call instead of 3 separate Python/PyTorch ops.
 *
 * Key optimizations:
 * - Single pass through logits for top-k + softmax
 * - No intermediate tensor allocation
 * - Numerically stable softmax
 * - Direct random sampling without torch.multinomial overhead
 */

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <numeric>

namespace nicto {

// ============================================================
// Fused top-k filter + softmax + sample
// ============================================================

// Filter logits to top-k, compute softmax, sample one token
int64_t fused_sample(
    const float* logits,      // [vocab_size]
    int64_t vocab_size,
    float temperature,
    int64_t top_k
) {
    // Clamp top_k
    if (top_k <= 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }

    // Apply temperature
    std::vector<float> buf(vocab_size);
    if (temperature != 1.0f && temperature > 0.0f) {
        float inv_temp = 1.0f / temperature;
        for (int64_t i = 0; i < vocab_size; i++) {
            buf[i] = logits[i] * inv_temp;
        }
    } else {
        std::memcpy(buf.data(), logits, vocab_size * sizeof(float));
    }

    // Find top-k threshold using partial sort
    if (top_k < vocab_size) {
        // Use nth_element for O(N) top-k instead of O(N log N) sort
        std::vector<int64_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);

        std::nth_element(
            indices.begin(), indices.begin() + top_k, indices.end(),
            [&buf](int64_t a, int64_t b) { return buf[a] > buf[b]; }
        );

        float threshold = buf[indices[top_k - 1]];

        // Mask below threshold
        for (int64_t i = 0; i < vocab_size; i++) {
            if (buf[i] < threshold) {
                buf[i] = -1e9f;
            }
        }
    }

    // Numerically stable softmax
    float max_val = buf[0];
    for (int64_t i = 1; i < vocab_size; i++) {
        if (buf[i] > max_val) max_val = buf[i];
    }

    float sum_exp = 0.0f;
    for (int64_t i = 0; i < vocab_size; i++) {
        buf[i] = std::exp(buf[i] - max_val);
        sum_exp += buf[i];
    }

    float inv_sum = 1.0f / sum_exp;
    for (int64_t i = 0; i < vocab_size; i++) {
        buf[i] *= inv_sum;
    }

    // Multinomial sample using CDF
    float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX + 1.0f);
    float cumsum = 0.0f;
    for (int64_t i = 0; i < vocab_size; i++) {
        cumsum += buf[i];
        if (cumsum >= r) {
            return i;
        }
    }
    return vocab_size - 1;
}

// ============================================================
// Batch sample: multiple logit vectors at once
// ============================================================
void fused_sample_batch(
    const float* logits,      // [batch * vocab_size]
    int64_t batch_size,
    int64_t vocab_size,
    float temperature,
    int64_t top_k,
    int64_t* out_tokens       // [batch] output
) {
    for (int64_t b = 0; b < batch_size; b++) {
        out_tokens[b] = fused_sample(
            logits + b * vocab_size,
            vocab_size,
            temperature,
            top_k
        );
    }
}

} // namespace nicto
