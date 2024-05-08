// Copyright (C) 2024  Ologan Ltd
// SPDX-License-Identifier: MIT
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// cuBLAS convenience type conversion helpers for API

#include <cublas_v2.h>

namespace gemm {

//
// cublasComputeType_t defaults for a given data type (i.e. same size)
//
template <typename T>
constexpr cublasComputeType_t cublasDefaultComputeType();

template <>
constexpr cublasComputeType_t cublasDefaultComputeType<float>() {
    return CUBLAS_COMPUTE_32F;
}

template <>
constexpr cublasComputeType_t cublasDefaultComputeType<double>() {
    return CUBLAS_COMPUTE_64F;
}

#if defined __half
template <>
constexpr cublasComputeType_t cublasDefaultComputeType<__half>() {
    return CUBLAS_COMPUTE_16F;
}
#endif

}
