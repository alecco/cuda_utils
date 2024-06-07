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

// cuda memory helper

#include <iostream>
#include <memory>
#include <source_location>
#include <format>
#include <functional>

namespace util {

// Class for managing CUDA device memory
template <typename T>
class CudaMemory {
    void deleter(T* p, const std::string& info) {
        if (p != nullptr) {
            cudaError_t err = cudaFree(p);
            if (err != cudaSuccess) {
                std::cerr << std::format("Failed to free device memory: {}\n",
                        std::string(cudaGetErrorString(err)), info);
                // warn but let it slide
            }
        }
    }
    std::unique_ptr<T, std::function<void(T*)>> ptr;
    size_t size;
    const std::string info;                 // debug info

public:
    // Called to allocate later
    explicit CudaMemory(const std::string info_ = "")
            : ptr(nullptr, [this, info = info_](T* p) { deleter(p, info); }), info(info_), size(0) { }

    // Allocate now
    explicit CudaMemory(size_t size, const std::string info_ = "", 
            const std::source_location& loc = std::source_location::current())
            : ptr(nullptr, [this, info = info_](T* p) { deleter(p, info); }), info(info_) {
        alloc(size, loc);
    }

    void alloc(const size_t size_, const std::source_location& loc = std::source_location::current()) {
        T* device_ptr;
        cudaError_t err = cudaMalloc(&device_ptr, size_ * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << std::format("Failed to allocate device memory: {}\n{}:{}\n",
                    std::string(cudaGetErrorString(err)), info, loc.file_name(), loc.line());
            std::exit(EXIT_FAILURE);
        }
        ptr.reset(device_ptr);   // free old pointed data, assign new pointer
        size = size_;
    }

    T* get() const {
        return ptr.get();
    }

    size_t len() const {
        return size;
    }

    size_t size_bytes() const {
        return size * sizeof(T);
    }
};
} // namespace util
