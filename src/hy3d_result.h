#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace hy3d {

template <typename T>
class Result {
public:
    static Result success(T value) {
        Result result;
        result.ok_ = true;
        result.value_ = std::move(value);
        return result;
    }

    static Result failure(std::string error) {
        Result result;
        result.ok_ = false;
        result.error_ = std::move(error);
        return result;
    }

    [[nodiscard]] bool ok() const {
        return ok_;
    }

    const T& value() const {
        if (!ok_) {
            throw std::logic_error("Result has no value");
        }
        return value_;
    }

    T& value() {
        if (!ok_) {
            throw std::logic_error("Result has no value");
        }
        return value_;
    }

    const std::string& error() const {
        if (ok_) {
            throw std::logic_error("Result has no error");
        }
        return error_;
    }

private:
    bool ok_ = false;
    T value_{};
    std::string error_;
};

} // namespace hy3d
