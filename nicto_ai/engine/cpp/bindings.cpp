/*
 * NICTO AI - C++ Engine pybind11 bindings
 *
 * Exposes C++ fast tokenizer, MoE, generation, and scan
 * functions to Python via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.cpp"
#include "moe.cpp"
#include "generate.cpp"
#include "scan.cpp"

namespace py = pybind11;

PYBIND11_MODULE(nicto_engine, m) {
    m.doc() = "NICTO AI fast C++ engines for tokenizer, MoE, generation, and SSM scan";

    // Tokenizer
    m.def("fast_encode", &nicto_engine::fast_encode,
          "Fast byte-level encode: UTF-8 string -> token IDs",
          py::arg("text"), py::arg("vocab_size"));

    m.def("fast_decode", &nicto_engine::fast_decode,
          "Fast byte-level decode: token IDs -> UTF-8 string",
          py::arg("tokens"));

    m.def("fast_batch_encode", &nicto_engine::fast_batch_encode,
          "Batch encode multiple strings to padded token matrix",
          py::arg("texts"), py::arg("vocab_size"), py::arg("max_len"));

    m.def("find_stop_token", &nicto_engine::find_stop_token,
          "Find stop token position in text (-1 if not found)",
          py::arg("text"));

    // MoE
    m.def("fused_topk", &nicto_engine::fused_topk路由,
          "Fused top-k routing: gate logits -> expert assignments + weights",
          py::arg("gate_logits"), py::arg("n_tokens"), py::arg("n_experts"),
          py::arg("k"), py::arg("expert_ids"), py::arg("weights"));

    m.def("compute_load_balance_loss", &nicto_engine::compute_load_balance_loss,
          "Vectorized load balance loss computation",
          py::arg("expert_ids"), py::arg("n_tokens"), py::arg("k"),
          py::arg("n_experts"));

    // Generation
    m.def("fused_sample", &nicto_engine::fused_sample,
          "Fused top-k + softmax + multinomial sampling from logits",
          py::arg("logits"), py::arg("vocab_size"), py::arg("temperature"),
          py::arg("top_k"));

    // Scan
    m.def("selective_scan", &nicto_engine::selective_scan,
          "Selective SSM scan (Mamba-style)",
          py::arg("x"), py::arg("dt"), py::arg("A"), py::arg("B"),
          py::arg("C"), py::arg("y"), py::arg("h"),
          py::arg("batch"), py::arg("seq_len"), py::arg("d_inner"),
          py::arg("d_state"));

    m.def("liquid_neuron_scan", &nicto_engine::liquid_neuron_scan,
          "Liquid neural network neuron step with Euler discretization",
          py::arg("x"), py::arg("h"), py::arg("y"),
          py::arg("batch"), py::arg("seq_len"), py::arg("dim"),
          py::arg("lambda"), py::arg("dt"));
}
