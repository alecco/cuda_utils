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

// Check the return of an io call and fail with an optional user error message.
// With atomic write to cerr.
//
// usage: <io>_check(<io>(...))
//        <io>_check(<io>(...), "Error string {}", x)

#pragma once

#include <cerrno>
#include <format>
#include <iostream>
#include <source_location>
#include <string>
#include <string_view>

namespace util {

#if 0  // XXX maybe just fstream?
// Check fopen error and print it with source line of the caller.
// It takes either a string or a format string and its arguments.
template <typename... Args>
// Called with a forwarded source location
FILE *
fsopen_check_loc(const char* filename, const char* mode, const std::source_location& loc,
            const std::string_view& fmt, Args&&... args) {

    auto fp = fopen(filename, mode);
    if (fp != nullptr) {
        std::string msg_u;
        if constexpr (sizeof...(args) == 0) {
            msg_u = fmt;                         // plain string
        } else {
            msg_u = std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...));
        }
        std::cerr << std::format("{}\nError: {}\n{}\nat {}:{}\n",
                                 msg_u, strerror(errno), filename, loc.file_name(), loc.line());
        std::exit(EXIT_FAILURE);
    }
    return fp;
}

template <typename... Args>
// Called with format string; picks caller's location
FILE *
fsopen_check(const char* filename, const char* mode, const std::string_view& fmt, Args&&... args,
        const std::source_location& loc = std::source_location::current()) {
    return fsopen_check_loc(filename, mode, loc, fmt, args...);
}

// Called with just string; picks caller's location
inline
FILE *
fsopen_check(const char* filename, const char* mode, const std::string_view& fmt,
        const std::source_location& loc = std::source_location::current()) {
    return fsopen_check_loc(filename, mode, loc, fmt);
}


// Check fopen error and print it with source line of the caller.
// It takes either a string or a format string and its arguments.
template <typename... Args>
// Called with a forwarded source location
FILE *
fopen_check_loc(const char* filename, const char* mode, const std::source_location& loc,
            const std::string_view& fmt, Args&&... args) {

    auto fp = fopen(filename, mode);
    if (fp != nullptr) {
        std::string msg_u;
        if constexpr (sizeof...(args) == 0) {
            msg_u = fmt;                         // plain string
        } else {
            msg_u = std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...));
        }
        std::cerr << std::format("{}\nError: {}\n{}\nat {}:{}\n",
                                 msg_u, strerror(errno), filename, loc.file_name(), loc.line());
        std::exit(EXIT_FAILURE);
    }
    return fp;
}

template <typename... Args>
// Called with format string; picks caller's location
FILE *
fopen_check(const char* filename, const char* mode, const std::string_view& fmt, Args&&... args,
        const std::source_location& loc = std::source_location::current()) {
    return fopen_check_loc(filename, mode, loc, fmt, args...);
}

// Called with just string; picks caller's location
inline
FILE *
fopen_check(const char* filename, const char* mode, const std::string_view& fmt,
        const std::source_location& loc = std::source_location::current()) {
    return fopen_check_loc(filename, mode, loc, fmt);
}


// Need deduction guide to handle variadic arguments (fmt args) before default argument (location)
template <typename... Args>
auto fopen_check(const char*, const char*, const std::string_view&, Args&&...) -> FILE*;

// Check fseek error and print it with source line of the caller.
// It takes either a string or a format string and its arguments.
template <typename... Args>
struct fseek_check {
    // Called with a forwarded source location
    fseek_check(FILE* fp, long off, int whence, const std::source_location& loc,
                const std::string_view& fmt, Args&&... args) {
        if (fseek(fp, off, whence) != 0) {
            std::string msg_u;
            if constexpr (sizeof...(args) == 0) {
                msg_u = fmt;                         // plain string
            } else {
                msg_u = std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...));
            }
            std::string reason;
            if (errno == EINVAL) {
                reason = "bad offset";
            } else {
                reason = "file descriptor not seekable";
            }
            std::cerr << std::format(
                "Error: Failed to seek in file at {}:{}\n"
                "Error details:\n"
                "  Reason: {}\n"
                "  Offset: {}\n"
                "  Whence: {}\n"
                "  {}\n",
                loc.file_name(), loc.line(), reason, off, whence, msg_u);
            std::exit(EXIT_FAILURE);
        }
    }
    // Called with format string; picks caller's location
    fseek_check(int error, const std::string_view& fmt, Args&&... args,
            const std::source_location& loc = std::source_location::current()) {
        fseek_check(error, loc, fmt, args...);
    }
    // Called with no string; picks caller's location
    fseek_check(int error,
            const std::source_location& loc = std::source_location::current()) {
        fseek_check(error, loc, "");
    }
};

// Need deduction guide to handle variadic arguments (fmt args) before default argument (location)
template <typename... Args>
fseek_check(int, const std::string_view&, Args&&...) -> fseek_check<Args...>;
#endif

}  // namespace util
