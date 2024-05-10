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

// wrappers for cuBLAS GEMM API

#include <util/cublas_helpers.h>
#include <util/cuda_helpers.h>
#include <source_location>
#include <cublas_v2.h>

namespace gemm {

// cuBLAS gemm wrapper
//
// check the table in https://docs.nvidia.com/cuda/cublas/#cublasgemmex
// and consider https://docs.nvidia.com/cuda/cublas/#tensor-core-usage
template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
inline cublasStatus_t
cublas_gemm(cublasHandle_t handle, cublasOperation_t a_op, cublasOperation_t b_op,
     int m, int n, int k,
     const Alpha* alpha,
     const TA* A, int ldA,
     const TB* B, int ldB,
     const Beta* beta,
     TC* C, int ldC,
     cublasComputeType_t computeType) {
    return cublasGemmEx(handle, a_op, b_op,
                        m, n, k,
                        &alpha, A, getCudaDataType<TA>(), ldA, B, getCudaDataType<TB>(), ldB,
                        &beta, C, getCudaDataType<TC>(), ldC,
                        computeType,
                        CUBLAS_GEMM_DEFAULT);
}

// Helper lambda (e.g. for benchmark)
template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
auto
cublas_func(cublasHandle_t handle, cublasOperation_t a_op, cublasOperation_t b_op,
     int m, int n, int k,
     const Alpha* alpha,
     const TA* A, int ldA,
     const TB* B, int ldB,
     const Beta* beta,
     TC* C, int ldC,
     cublasComputeType_t computeType,
     const std::source_location& loc = std::source_location::current()
     ) {
    return [=]() -> void {
        cublas_check(
                cublas_gemm(handle, a_op, b_op, m, n, k,
                            alpha, A, ldA, B, ldB, beta, C, ldC, computeType),
                loc,               // pass through caller's source location
                "gemm error{}");
    };
}
} // namespace gemm
