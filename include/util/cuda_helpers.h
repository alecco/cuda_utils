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

// cuda convenience type conversion helpers for API

#include <library_types.h>

namespace gemm {

//
// cudaDataType helpers
//
template <typename T>
constexpr cudaDataType_t getCudaDataType();

template <>
constexpr cudaDataType_t getCudaDataType<float>() {
    return CUDA_R_32F;
}

template <>
constexpr cudaDataType_t getCudaDataType<double>() {
    return CUDA_R_64F;
}

#ifdef __half
template <>
constexpr cudaDataType_t getCudaDataType<__half>() {
    return CUDA_R_16F;
}
#endif

#ifdef __nv_bfloat16
template <>
constexpr cudaDataType_t getCudaDataType<__nv_bfloat16>() {
    return CUDA_R_16BF;
}
#endif

#ifdef __NV_E4M3
template <>
constexpr cudaDataType_t getCudaDataType<__NV_E4M3>() {
    return CUDA_R_8F_E4M3;
}
#endif

#ifdef __NV_E5M2
template <>
constexpr cudaDataType_t getCudaDataType<__NV_E5M2>() {
    return CUDA_R_8F_E5M2;
}
#endif

#ifdef int8_t
template <>
constexpr cudaDataType_t getCudaDataType<int8_t>() {
    return CUDA_R_8I;
}
#endif

}
