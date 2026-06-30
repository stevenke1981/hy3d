#include "hy3d_result.h"

#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    auto result = hy3d::Result<std::vector<int>>::success({1, 2, 3});
    auto values = result.take_value();
    if (values != std::vector<int>({1, 2, 3})) {
        std::cerr << "take_value did not return the stored value\n";
        return 1;
    }

    auto failure = hy3d::Result<std::vector<int>>::failure("failed");
    try {
        (void)failure.take_value();
        std::cerr << "take_value accepted a failed Result\n";
        return 1;
    } catch (const std::logic_error&) {
    }

    return 0;
}
