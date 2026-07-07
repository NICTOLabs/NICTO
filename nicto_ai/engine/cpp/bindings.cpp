/*
 * NICTO AI - C++ Engine pybind11 bindings
 *
 * Exposes all C++ fast functions to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.h"
#include "sampler.h"
#include "generator.h"

namespace py = pybind11;

PYBIND11_MODULE(nicto_engine, m) {
    m.doc() = "NICTO AI fast C++ engines";

    // ---- Tokenizer ----
    m.def("encode", &nicto::encode,
          "UTF-8 text -> token IDs",
          py::arg("text"), py::arg("vocab_size") = 32000);

    m.def("decode", [](const std::vector<int64_t>& tokens) -> std::string {
        return nicto::decode(tokens.data(), tokens.size());
    }, "Token IDs -> UTF-8 text", py::arg("tokens"));

    m.def("batch_encode", &nicto::batch_encode,
          "Batch encode strings to padded token matrix",
          py::arg("texts"), py::arg("vocab_size") = 32000, py::arg("max_len") = 2048);

    m.def("find_stop", [](const std::string& text) -> int64_t {
        return nicto::find_stop(text.data(), text.size());
    }, "Find stop token position (-1 if not found)", py::arg("text"));

    // ---- Sampler ----
    m.def("sample", &nicto::fused_sample,
          "Fused top-k + softmax + multinomial sample",
          py::arg("logits"), py::arg("vocab_size"),
          py::arg("temperature") = 0.8f, py::arg("top_k") = 50);

    m.def("sample_batch", &nicto::fused_sample_batch,
          "Batch fused sampling",
          py::arg("logits"), py::arg("batch_size"), py::arg("vocab_size"),
          py::arg("temperature") = 0.8f, py::arg("top_k") = 50);

    // ---- Generator ----
    m.def("find_stop_in_tokens", &nicto::find_stop_in_tokens,
          "Find double-newline position in token array",
          py::arg("tokens"), py::arg("n_tokens"));
}
