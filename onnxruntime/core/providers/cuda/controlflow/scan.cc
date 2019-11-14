// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/controlflow/scan.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/transpose.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  8, 8,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  Scan<8>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  9, 10,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_KERNEL_EX(Scan,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                        Scan<9>);

Status Scan<8>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<8>::Compute(ctx);
  return status;
}

Status Scan<9>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<9>::Compute(ctx);
  return status;
}

Status Scan<8>::TransposeOutput(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) const {
  ORT_NOT_IMPLEMENTED("Scan<8> spec does not support transpose of output. This should never be called.");
}

Status Scan<9>::TransposeOutput(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) const {
  const OpKernelInfo& info = OpKernel::Info();
  return cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, input, output);
}

}  // namespace cuda
}  // namespace onnxruntime
