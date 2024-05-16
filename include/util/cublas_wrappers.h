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

// wrappers for cuBLAS

#include <util/cublas_helpers.h>
#include <util/cuda_helpers.h>
#include <util/cublas_check.h>
#include <source_location>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cublas_api.h>

namespace util {

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

/// XXX
/// D = alpha*(A*B) + beta*(C)
/// C can be nullptr for in-place compute
/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle

// from cublasLtMatmul() docs:
//      To use FP8 kernels, the following set of requirements must be satisfied:
//          All matrix pointers must be 16-byte aligned.
//          A must be transposed and B non-transposed (The “TN” format).
//          The compute type must be CUBLAS_COMPUTE_32F.
//          The scale type must be CUDA_R_32F.
//      workspace size: 32 MB for Hopper+, 4MB Ampere-
template <typename InType, typename OutType = InType, typename ComputeType = OutType>
void
LtMatmul(cublasLtHandle_t ltHandle, int m, int n, int k,
         const float*    alpha,            // host
         const float*    a_scale,          // device
         const InType*   A,                // device
         int             lda,
         const float*    b_scale,          // device
         const InType*   B,                // device
         int             ldb,
         const InType*   C,                // device
         const float*    c_scale,          // device
         InType *        D,                // device
         int             ldc,
         const float*    d_scale,          // device
         float*          amax_d,           // device
         void*           workspace,        // device, must be 16b aligned
         size_t          workspaceSize,
         cudaStream_t    stream) {

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    float beta = 0.0; // Can be non-zero starting from 12.0

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    cublas_check(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // Required
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // set scaling factors
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    cublas_check(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    cublas_check(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    cublas_check(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    cublas_check(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc)); // XXX does this matter?
    cublas_check(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    cublas_check(cublasLtMatmulPreferenceCreate(&preference));
    cublas_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    cublas_check(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        cublas_check(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    cublas_check(cublasLtMatmul(ltHandle,
                                operationDesc,
                                alpha,
                                A,
                                Adesc,
                                B,
                                Bdesc,
                                &beta,
                                nullptr,    // no prev C, just D = alpha * A x B + bias (?)
                                Cdesc,
                                D,
                                Ddesc,
                                &heuristicResult.algo,
                                workspace,
                                workspaceSize,
                                0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) cublas_check(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) cublas_check(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) cublas_check(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) cublas_check(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) cublas_check(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) cublas_check(cublasLtMatmulDescDestroy(operationDesc));
}

} // namespace util
