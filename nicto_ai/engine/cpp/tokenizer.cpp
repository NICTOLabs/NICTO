/*
 * NICTO AI - Fast Byte-Level Tokenizer (C++)
 * 
 * Encodes UTF-8 text to token IDs and decodes token IDs back to text.
 * Uses direct byte-to-token mapping (same as NICTO's training).
 *
 * Compile with: pybind11 + torch C++ extensions
 */

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace nicto_engine {

// Fast byte-level encode: UTF-8 string -> token IDs
// Each byte maps to token_id = byte % vocab_size
std::vector<int64_t> fast_encode(const std::string& text, int64_t vocab_size) {
    std::vector<int64_t> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int64_t>(c) % vocab_size);
    }
    return tokens;
}

// Fast byte-level decode: token IDs -> UTF-8 string
// Each token maps back to byte = token % 256
std::string fast_decode(const std::vector<int64_t>& tokens) {
    std::string result;
    result.reserve(tokens.size());
    for (int64_t t : tokens) {
        result.push_back(static_cast<char>(t % 256));
    }
    return result;
}

// Batch encode: multiple strings -> padded token matrix
// Returns tokens matrix [batch, max_len] and lengths vector
struct BatchEncodeResult {
    std::vector<std::vector<int64_t>> tokens;
    std::vector<int64_t> lengths;
};

BatchEncodeResult fast_batch_encode(
    const std::vector<std::string>& texts,
    int64_t vocab_size,
    int64_t max_len
) {
    BatchEncodeResult result;
    result.tokens.resize(texts.size());
    result.lengths.resize(texts.size());

    for (size_t i = 0; i < texts.size(); i++) {
        auto& enc = fast_encode(texts[i], vocab_size);
        result.lengths[i] = std::min(static_cast<int64_t>(enc.size()), max_len);
        result.tokens[i].resize(max_len, 0);
        for (int64_t j = 0; j < result.lengths[i]; j++) {
            result.tokens[i][j] = enc[j];
        }
    }
    return result;
}

// Find stop token position in text (returns -1 if not found)
int find_stop_token(const std::string& text) {
    // Check for common stop sequences
    const char* stops[] = {"\n\n", "Assistant:", "User:", "System:"};
    int min_pos = -1;

    for (const char* stop : stops) {
        size_t pos = text.find(stop);
        if (pos != std::string::npos) {
            int ipos = static_cast<int>(pos);
            if (min_pos == -1 || ipos < min_pos) {
                min_pos = ipos;
            }
        }
    }
    return min_pos;
}

} // namespace nicto_engine
