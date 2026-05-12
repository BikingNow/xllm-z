/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kCompileMaxNumSeqs = 256;
constexpr int32_t kCompileMaxNumCacheSlots = 1024;
constexpr int64_t kTokenPadding = 64;

class TileLangFusedSigmoidGatingDeltaRuleWrapperTest
    : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct FusedSigmoidGatingDeltaRuleTestCase {
  std::string name;
  std::vector<int64_t> seqlens;
  int64_t nk;
  int64_t nv;
  int64_t dk;
  int64_t dv;
  int64_t seed;
  float softplus_beta = 1.0F;
};

std::tuple<torch::Tensor, torch::Tensor>
torch_fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    float softplus_beta = 1.0F) {
  namespace F = torch::nn::functional;

  const auto total_tokens = query.size(1);
  const auto nk = query.size(2);
  const auto dk = query.size(3);
  const auto nv = value.size(2);
  const auto dv = value.size(3);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(dk));
  const float l2_norm_eps = 1e-12F;
  const int64_t v_per_k = nv / nk;
  const auto fp32_opts = query.options().dtype(torch::kFloat32);

  auto state = torch::zeros({num_seqs, nv, dk, dv}, fp32_opts);
  for (int64_t i = 0; i < num_seqs; ++i) {
    auto state_idx = ssm_state_indices[i].item<int64_t>();
    if (state_idx >= 0) {
      state[i] = init_state[state_idx].to(torch::kFloat32);
    }
  }

  auto out = torch::empty({1, total_tokens, nv, dv}, fp32_opts);
  auto exp_A = torch::exp(A_log.to(torch::kFloat32));

  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    int64_t seq_start = cu_seqlens[seq_idx].item<int64_t>();
    int64_t seq_end = cu_seqlens[seq_idx + 1].item<int64_t>();
    for (int64_t v_head_idx = 0; v_head_idx < nv; ++v_head_idx) {
      auto h = state[seq_idx][v_head_idx];
      int64_t k_head_idx = v_head_idx / v_per_k;
      for (int64_t t = 0; t < seq_end - seq_start; ++t) {
        int64_t token_idx = seq_start + t;
        auto q_t = query[0][token_idx][k_head_idx].to(torch::kFloat32);
        auto k_t = key[0][token_idx][k_head_idx].to(torch::kFloat32);
        auto v_t = value[0][token_idx][v_head_idx].to(torch::kFloat32);

        // L2 norm.
        q_t = q_t / torch::sqrt(q_t.pow(2).sum() + l2_norm_eps);
        k_t = k_t / torch::sqrt(k_t.pow(2).sum() + l2_norm_eps);

        // Softplus.
        auto x = a[token_idx][v_head_idx].to(torch::kFloat32) +
                 dt_bias[v_head_idx].to(torch::kFloat32);
        auto beta_x = softplus_beta * x;
        auto sp = torch::where(
            beta_x > 20.0F,
            x,
            torch::log1p(torch::exp(beta_x)) / softplus_beta);

        h = h * torch::exp(-exp_A[v_head_idx] * sp);
        auto pred = torch::matmul(k_t, h);
        auto gate = torch::sigmoid(
            beta[token_idx][v_head_idx].to(torch::kFloat32));
        h = h + torch::outer(k_t, (v_t - pred) * gate);
        out[0][token_idx][v_head_idx] = torch::matmul(q_t * scale, h);
      }
      state[seq_idx][v_head_idx] = h;
    }
  }

  return {out.to(query.dtype()), state.to(init_state.dtype())};
}

torch::Tensor pad_to_size(const torch::Tensor& t,
                          int64_t target_dim0,
                          int64_t fill_value) {
  if (t.size(0) >= target_dim0) {
    return t.slice(0, 0, target_dim0);
  }
  auto pad_shape = t.sizes().vec();
  pad_shape[0] = target_dim0 - t.size(0);
  auto padding = torch::full(
      pad_shape, fill_value,
      torch::TensorOptions().dtype(t.dtype()).device(t.device()));
  return torch::cat({t, padding}, /*dim=*/0);
}

void run_fused_sigmoid_gating_delta_rule_case(
    const FusedSigmoidGatingDeltaRuleTestCase& test_case) {
  const auto device = torch::Device("npu:0");
  torch::manual_seed(test_case.seed);

  const int64_t num_seqs = static_cast<int64_t>(test_case.seqlens.size());
  ASSERT_GT(num_seqs, 0);
  ASSERT_LE(num_seqs, kCompileMaxNumSeqs);

  int64_t total_tokens = 0;
  std::vector<int32_t> cu_seqlens_vec;
  cu_seqlens_vec.push_back(0);
  for (auto len : test_case.seqlens) {
    total_tokens += len;
    cu_seqlens_vec.push_back(static_cast<int32_t>(total_tokens));
  }

  const auto fp16_opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(device);
  const auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  const int64_t padded_tokens = total_tokens + kTokenPadding;
  const int64_t nk = test_case.nk;
  const int64_t nv = test_case.nv;
  const int64_t dk = test_case.dk;
  const int64_t dv = test_case.dv;

  auto A_log = torch::randn({nv}, fp16_opts);
  auto a = torch::randn({padded_tokens, nv}, fp16_opts);
  auto dt_bias = torch::randn({nv}, fp16_opts);
  auto query = torch::randn({padded_tokens, nk, dk}, fp16_opts);
  auto key = torch::randn({padded_tokens, nk, dk}, fp16_opts);
  auto value = torch::randn({padded_tokens, nv, dv}, fp16_opts);
  auto beta = torch::randn({padded_tokens, nv}, fp16_opts);

  // Build init_state with compile-time max cache slots.
  int64_t num_cache_slots = num_seqs * 2;
  ASSERT_LE(num_cache_slots, kCompileMaxNumCacheSlots);
  auto init_state_actual =
      torch::randn({num_cache_slots, nv, dk, dv}, fp16_opts);
  auto init_state = pad_to_size(
      init_state_actual, kCompileMaxNumCacheSlots, /*fill_value=*/0);

  // Build ssm_state_indices and pad to compile max.
  auto ssm_state_indices_actual =
      torch::arange(num_seqs, i32_opts);  // 0, 1, 2, ...
  auto ssm_state_indices = pad_to_size(
      ssm_state_indices_actual, kCompileMaxNumSeqs, /*fill_value=*/-1);

  // Build cu_seqlens and pad to compile max + 1.
  auto cu_seqlens_actual = torch::tensor(cu_seqlens_vec, i32_opts);
  auto cu_seqlens = pad_to_size(
      cu_seqlens_actual,
      kCompileMaxNumSeqs + 1,
      /*fill_value=*/cu_seqlens_vec.back());

  // PyTorch reference (uses unpadded data).
  auto query_ref = query.slice(0, 0, padded_tokens).unsqueeze(0);
  auto key_ref = key.slice(0, 0, padded_tokens).unsqueeze(0);
  auto value_ref = value.slice(0, 0, padded_tokens).unsqueeze(0);
  auto a_ref = a.slice(0, 0, padded_tokens);
  auto beta_ref = beta.slice(0, 0, padded_tokens);
  auto init_state_ref = init_state.slice(0, 0, num_cache_slots);

  auto [out_ref, final_state_ref] = torch_fused_sigmoid_gating_delta_rule(
      A_log.slice(0, 0, nv),
      a_ref,
      dt_bias.slice(0, 0, nv),
      query_ref,
      key_ref,
      value_ref,
      beta_ref,
      init_state_ref,
      ssm_state_indices_actual,
      cu_seqlens_actual,
      test_case.softplus_beta);

  // TileLang kernel.
  auto [out_out, final_state_out] = fused_sigmoid_gating_delta_rule(
      A_log, a, dt_bias, query, key, value, beta, init_state,
      ssm_state_indices, cu_seqlens);

  // Slice outputs back to actual sizes.
  auto out_sliced = out_out.slice(0, 0, total_tokens).unsqueeze(0);
  auto final_state_sliced = final_state_out.slice(0, 0, num_seqs);

  // Compare.
  auto out_max_diff =
      (out_sliced.to(torch::kFloat32) - out_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();
  auto final_state_max_diff =
      (final_state_sliced.to(torch::kFloat32) -
       final_state_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();

  EXPECT_TRUE(torch::allclose(out_sliced, out_ref, /*rtol=*/2e-2, /*atol=*/2e-2))
      << "out mismatch, max_diff=" << out_max_diff;
  EXPECT_TRUE(torch::allclose(final_state_sliced, final_state_ref,
                              /*rtol=*/2e-2, /*atol=*/2e-2))
      << "final_state mismatch, max_diff=" << final_state_max_diff;
}

TEST_F(TileLangFusedSigmoidGatingDeltaRuleWrapperTest,
       MatchesTorchReference) {
  const std::vector<FusedSigmoidGatingDeltaRuleTestCase> cases = {
      {
          .name = "small_4x8_d128",
          .seqlens = {4, 8, 6, 3},
          .nk = 4,
          .nv = 8,
          .dk = 128,
          .dv = 128,
          .seed = 101,
      },
      {
          .name = "medium_16x32_d128",
          .seqlens = {4, 8, 4, 8, 4, 8, 4, 8},
          .nk = 16,
          .nv = 32,
          .dk = 128,
          .dv = 128,
          .seed = 102,
      },
      {
          .name = "more_seqs_16x32_d128",
          .seqlens = {3, 5, 7, 2, 4, 6, 8, 3,
                       5, 7, 2, 4, 6, 8, 3, 5},
          .nk = 16,
          .nv = 32,
          .dk = 128,
          .dv = 128,
          .seed = 103,
      },
      {
          .name = "long_seqs_4x8_d128",
          .seqlens = {128, 64, 32, 16},
          .nk = 4,
          .nv = 8,
          .dk = 128,
          .dv = 128,
          .seed = 104,
      },
      {
          .name = "single_seq_4x8_d128",
          .seqlens = {256},
          .nk = 4,
          .nv = 8,
          .dk = 128,
          .dv = 128,
          .seed = 105,
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    run_fused_sigmoid_gating_delta_rule_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
