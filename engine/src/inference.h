#pragma once

#include "action.h"
#include "encoder.h"

#include <memory>
#include <string>

namespace pt {

// LibTorch-backed inference wrapper around a TorchScript-exported DMCNet.
//
// The loaded module must expose `forward(x: Tensor, a: Tensor) -> Tensor`
// where x is (B, X_DIM) and a is (B, A_DIM), returning (B,). This matches the
// `DMCNet` in trainer/dmc/models.py after torch.jit.script (see
// trainer/export.py for the producer).
//
// A `ModelInference` is NOT thread-safe: scratch buffers are reused across
// calls. Create one per actor/thread.
class ModelInference {
public:
    // Loads `path` as a TorchScript module. `device_str` is any torch device
    // string LibTorch understands ("cpu", "cuda", "cuda:0", ...).
    explicit ModelInference(const std::string& path,
                            const std::string& device_str = "cpu");
    ~ModelInference();
    ModelInference(ModelInference&&) noexcept;
    ModelInference& operator=(ModelInference&&) noexcept;
    ModelInference(const ModelInference&)            = delete;
    ModelInference& operator=(const ModelInference&) = delete;

    // Score every legal action for `state`. Writes one Q-value per legal
    // action into `out_q` (must have room for state.a.size() floats).
    // Returns the argmax index into [0, state.a.size()).
    int score_and_argmax(const EncodedState& state, float* out_q) const;

    // Convenience: score + argmax and return the chosen ActionType.
    ActionType pick(const EncodedState& state) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pt
