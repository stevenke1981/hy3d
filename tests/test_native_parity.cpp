#include "hy3d_math.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_native_parity <fixture.csv>\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "failed to open reference fixture\n";
        return 2;
    }

    std::unordered_map<std::string, std::vector<float>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string name;
        std::getline(stream, name, ',');
        std::string value;
        while (std::getline(stream, value, ',')) {
            rows[name].push_back(std::stof(value));
        }
    }

    const auto output = hy3d::scaled_dot_product_attention(rows["q"], rows["k"], rows["v"], 2, 2, 1, 2);
    if (!output.ok() || output.value().size() != rows["expected"].size()) {
        std::cerr << "native attention failed against NumPy fixture\n";
        return 1;
    }
    for (std::size_t i = 0; i < output.value().size(); ++i) {
        if (!std::isfinite(output.value()[i]) || std::fabs(output.value()[i] - rows["expected"][i]) > 1e-5f) {
            std::cerr << "parity mismatch at " << i << ": actual=" << output.value()[i]
                      << " expected=" << rows["expected"][i] << "\n";
            return 1;
        }
    }
    return 0;
}
