/*
 * NICTO AI - C++ Engine pybind11 bindings
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.h"
#include "sampler.h"
#include "generator.h"

namespace py = pybind11;

PYBIND11_MODULE(nicto_engine, m) {
    m.doc() = "NICTO AI fast C++ engines";

    // Tokenizer
    m.def("encode", [](const std::string& text, int64_t vocab_size) {
        auto tokens = nicto::encode(text, vocab_size);
        return tokens;
    }, "UTF-8 text -> token IDs",
       py::arg("text"), py::arg("vocab_size") = 32000);

    m.def("decode", [](const std::vector<int64_t>& tokens) -> std::string {
        return nicto::decode(tokens.data(), tokens.size());
    }, "Token IDs -> UTF-8 text", py::arg("tokens"));

    m.def("find_stop", [](const std::string& text) -> int64_t {
        return nicto::find_stop(text.data(), text.size());
    }, "Find stop token position (-1 if not found)", py::arg("text"));

    // Sampler
    m.def("sample", [](const std::vector<double>& logits, int64_t vocab_size,
                       float temperature, int64_t top_k) -> int64_t {
        std::vector<float> flogits(logits.begin(), logits.end());
        return nicto::fused_sample(flogits.data(), vocab_size, temperature, top_k);
    }, "Fused top-k + softmax + multinomial sample",
       py::arg("logits"), py::arg("vocab_size"),
       py::arg("temperature") = 0.8, py::arg("top_k") = 50);

    m.def("sample_float", [](const std::vector<float>& logits, int64_t vocab_size,
                             float temperature, int64_t top_k) -> int64_t {
        return nicto::fused_sample(logits.data(), vocab_size, temperature, top_k);
    }, "Fused sample (float input)",
       py::arg("logits"), py::arg("vocab_size"),
       py::arg("temperature") = 0.8f, py::arg("top_k") = 50);

    // Generator
    m.def("find_stop_in_tokens", [](const std::vector<int64_t>& tokens) -> int64_t {
        return nicto::find_stop_in_tokens(tokens.data(), tokens.size());
    }, "Find double-newline position in token array", py::arg("tokens"));
}
