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

//
// Type to string conversion helpers
//
// usage: type_name<type>()

#pragma once

// NOTE: using std::string because std::make_format_args() does not like std::string_view atm
#include <string>

namespace gemm {
template <typename T>
constexpr const std::string type_name();

template <>
constexpr const std::string type_name<float>() {
    return "fp32";
}

template <>
constexpr const std::string type_name<double>() {
    return "fp64";
}

#ifdef __half
template <>
constexpr const std::string type_name<__half>() {
    return "fp16";
}
#endif

#ifdef __nv_bfloat16
template <>
constexpr const std::string type_name<__nv_bfloat16>() {
    return "bf16";
}
#endif

#ifdef __NV_E4M3
template <>
constexpr const std::string type_name<__NV_E4M3>() {
    return "fp8-e4m3";
}
#endif

#ifdef __NV_E5M2
template <>
constexpr const std::string type_name<__NV_E5M2>() {
    return "fp8-e5m2";
}
#endif

#ifdef int8_t
template <>
constexpr const std::string type_name<int8_t>() {
    return "int8";
}
#endif

} // namespace gemm
