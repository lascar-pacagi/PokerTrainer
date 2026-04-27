#include "inference.h"

#include <torch/script.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace pt {

struct ModelInference::Impl {
    torch::jit::script::Module module;
    torch::Device              device;
    // Reused scratch for packing (n_legal, A_DIM) action features. Avoids
    // per-call allocation on the hot path.
    std::vector<float>         a_flat;

    Impl(const std::string& path, const std::string& device_str)
        : module(torch::jit::load(path)), device(device_str) {}
};

ModelInference::ModelInference(const std::string& path,
                               const std::string& device_str)
    : impl_(std::make_unique<Impl>(path, device_str)) {
    impl_->module.to(impl_->device);
    impl_->module.train(false);  // inference mode (no dropout/BN to toggle, belt+braces).
}

ModelInference::~ModelInference() = default;
ModelInference::ModelInference(ModelInference&&) noexcept            = default;
ModelInference& ModelInference::operator=(ModelInference&&) noexcept = default;

int ModelInference::score_and_argmax(const EncodedState& state,
                                     float* out_q) const {
    const int n = static_cast<int>(state.a.size());
    if (n <= 0) {
        throw std::runtime_error("ModelInference::score_and_argmax: no legal actions");
    }

    impl_->a_flat.resize(static_cast<std::size_t>(n) * A_DIM);
    for (int i = 0; i < n; ++i) {
        std::memcpy(&impl_->a_flat[static_cast<std::size_t>(i) * A_DIM],
                    state.a[i].data(), A_DIM * sizeof(float));
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);

    // x lives in EncodedState as a fixed-size array. from_blob + expand gives
    // a (n, X_DIM) view without copying X_DIM floats n times.
    torch::Tensor x = torch::from_blob(
        const_cast<float*>(state.x.data()), {1, X_DIM}, opts).expand({n, X_DIM});
    torch::Tensor a = torch::from_blob(
        impl_->a_flat.data(), {n, A_DIM}, opts);

    if (impl_->device.type() != torch::kCPU) {
        x = x.to(impl_->device);
        a = a.to(impl_->device);
    }

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x);
    inputs.emplace_back(a);
    at::Tensor v = impl_->module.forward(inputs).toTensor();
    if (impl_->device.type() != torch::kCPU) {
        v = v.to(torch::kCPU);
    }
    v = v.contiguous();

    const float* v_data = v.data_ptr<float>();
    int   best  = 0;
    float best_v = v_data[0];
    out_q[0] = v_data[0];
    for (int i = 1; i < n; ++i) {
        out_q[i] = v_data[i];
        if (v_data[i] > best_v) {
            best_v = v_data[i];
            best   = i;
        }
    }
    return best;
}

ActionType ModelInference::pick(const EncodedState& state) const {
    std::vector<float> q(state.a.size());
    const int idx = score_and_argmax(state, q.data());
    return state.legal[idx];
}

}  // namespace pt
