/*
 * NICTO AI - Fast Byte-Level Tokenizer (C++)
 *
 * Encodes UTF-8 text to token IDs and decodes token IDs back to text.
 * Uses direct byte-to-token mapping: token_id = byte % vocab_size.
 *
 * Key optimizations over Python:
 * - Single-pass encode (no list comprehension)
 * - Direct memory writes (no intermediate lists)
 * - SIMD-friendly memory layout
 * - Batch encode with pre-allocated output buffer
 */

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace nicto {

// ============================================================
// Single string encode: UTF-8 text -> token IDs
// ============================================================
void encode_into(
    const char* text,
    size_t text_len,
    int64_t* out_tokens,
    size_t max_tokens,
    int64_t vocab_size,
    size_t& out_len
) {
    out_len = 0;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text);
    for (size_t i = 0; i < text_len && out_len < max_tokens; i++) {
        out_tokens[out_len++] = static_cast<int64_t>(bytes[i]) % vocab_size;
    }
}

std::vector<int64_t> encode(const std::string& text, int64_t vocab_size) {
    std::vector<int64_t> tokens(text.size());
    size_t len = 0;
    encode_into(text.data(), text.size(), tokens.data(), tokens.size(), vocab_size, len);
    tokens.resize(len);
    return tokens;
}

// ============================================================
// Decode: token IDs -> UTF-8 string
// ============================================================
void decode_into(
    const int64_t* tokens,
    size_t n_tokens,
    char* out_text,
    size_t max_text,
    size_t& out_len
) {
    out_len = 0;
    for (size_t i = 0; i < n_tokens && out_len < max_text; i++) {
        out_text[out_len++] = static_cast<char>(tokens[i] % 256);
    }
}

std::string decode(const int64_t* tokens, size_t n_tokens) {
    std::string result(n_tokens, '\0');
    size_t len = 0;
    decode_into(tokens, n_tokens, &result[0], n_tokens, len);
    result.resize(len);
    return result;
}

// ============================================================
// Batch encode: multiple strings -> padded token matrix
// ============================================================
struct BatchResult {
    std::vector<int64_t> tokens;  // [batch * max_len] flattened
    std::vector<int64_t> lengths; // [batch]
    int64_t batch_size;
    int64_t max_len;
};

BatchResult batch_encode(
    const std::vector<std::string>& texts,
    int64_t vocab_size,
    int64_t max_len
) {
    BatchResult res;
    res.batch_size = texts.size();
    res.max_len = max_len;
    res.tokens.resize(texts.size() * max_len, 0);
    res.lengths.resize(texts.size());

    for (size_t i = 0; i < texts.size(); i++) {
        size_t enc_len = 0;
        encode_into(
            texts[i].data(), texts[i].size(),
            &res.tokens[i * max_len], max_len,
            vocab_size, enc_len
        );
        res.lengths[i] = enc_len;
    }
    return res;
}

// ============================================================
// Find stop token position
// ============================================================
int64_t find_stop(const char* text, size_t text_len) {
    int64_t min_pos = -1;

    const char* stops[] = {"\n\n", "Assistant:", "User:", "System:"};
    for (const char* stop : stops) {
        size_t stop_len = strlen(stop);
        if (stop_len == 0 || stop_len > text_len) continue;

        for (size_t i = 0; i <= text_len - stop_len; i++) {
            if (memcmp(text + i, stop, stop_len) == 0) {
                if (min_pos == -1 || static_cast<int64_t>(i) < min_pos) {
                    min_pos = static_cast<int64_t>(i);
                }
                break;
            }
        }
    }
    return min_pos;
}

} // namespace nicto
