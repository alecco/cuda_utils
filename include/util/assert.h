
// Copyright (C) 2024  Ologan Ltd
//
// SPDX-License-Identifier: AGPL-3.0
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Formatted assert
//
#pragma once

#include <format>
#include <iostream>
#include <source_location>
#include <string_view>

namespace util {

template <typename... Args>
struct assertf {
    assertf(bool condition, const std::source_location& loc, const std::string_view& fmt, Args&&... args) {
        if (!condition) {
            std::string msg_u;
            if constexpr (sizeof...(args) == 0) {
                msg_u = fmt;                         // plain string
            } else {
                msg_u = std::vformat(fmt, std::make_format_args(args...));
            }
            std::cerr << std::format("[ASSERT FAIL]: {}\n at {}:{}\n", msg_u, loc.file_name(), loc.line());
            exit(EXIT_FAILURE);
        }
    }
    
    // Called with format string; picks caller's location
    assertf(bool condition, const std::string_view& fmt, Args&&... args,
            const std::source_location& loc = std::source_location::current()) {
        assertf(condition, loc, fmt, std::forward<Args>(args)...);
    }
    
    // Called with no string; picks caller's location
    assertf(bool condition, const std::source_location& loc = std::source_location::current()) {
        assertf(condition, loc, "(unspecified)");
    }
};

// Need deduction guide to handle variadic arguments (fmt args) before default argument (location)
template <typename... Args>
assertf(bool, const std::string_view&, Args&&...) -> assertf<Args...>;

} // namespace util
