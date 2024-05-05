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
// Check the return of a cuBLAS call and fail with an optional user error message.
// With atomic write to cerr.
//
// usage: cublas_check(cublasCall())
//        cublas_check(cublasCall(), "Error string {}", x)

#pragma once

#include <cublas_v2.h>
#include <format>
#include <iostream>
#include <source_location>
#include <string_view>

namespace gemm {

// Check if a cuBLAS status is error and print line of the caller
// It takes either a string or a format string and its arguments.
template <typename... Args>
struct cublas_check {
    cublas_check(cublasStatus_t status, const std::string_view& fmt, Args&&... args, const std::source_location& loc = std::source_location::current()) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::string msg_u;
            if constexpr (sizeof...(args) == 0) {
                msg_u = fmt;
            } else {
                msg_u = std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...));
            }
            std::cerr << std::format("[cuBLAS ERROR]: {} at {}:{}\n    {}\n",
                   cublasGetStatusString(status), loc.file_name(), loc.line(), msg_u);
        }
    }
};

// Need deduction guide to handle variadic arguments (fmt args) before default argument (location)
template <typename... Args>
cublas_check(cublasStatus_t, const std::string_view&, Args&&...) -> cublas_check<Args...>;

}  // namespace gemm
