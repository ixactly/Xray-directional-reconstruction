//
// Created by tomokimori on 23/02/17.
//

#include "moire.cuh"
#include "Volume.h"
#include "Params.h"
#include "cmath"

void calcSinFittingLimited(const Volume<float> ct[4], Volume<float> out[3], int size_x, int size_y, int size_z) {
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                const float I[4] = {ct[0](x, y, z), ct[1](x, y, z), ct[2](x, y, z), ct[3](x, y, z)};
                const float a = (I[0] + I[1] + I[2] + I[3]) / 4.0f;
                const float b = std::sqrt((I[0] - I[2]) * (I[0] - I[2]) + (I[1] - I[3]) * (I[1] - I[3])) / 4.0f;
                float phi = std::atan2(I[1] - I[3], I[0] - I[2]) / 2.0f;

                // float phi = std::atan2(-(y - size_y / 2.0f), x - size_x / 2.0f);
                phi = (phi >= 0.0f) ? phi : (float) M_PI + phi;
                float r = std::sqrt(
                        (x - size_x / 2.0f) * (x - size_x / 2.0f) + (y - size_y / 2.0f) * (y - size_y / 2.0f));
                // std::cout << "(x, y, z): " << x << ", " << y << ", " << z << std::endl;
                // std::cout << I[0] << ", " << I[1] << ", " << I[2] << ", " << I[3] << ", " << I[4] << std::endl;
                // std::cout << ct[3](x, y, z) << std::endl;
                // r, g, b

                if (0.0f <= phi && phi <= M_PI / 4.0f) {
                    out[0](x, y, z) = -phi * 4.0f / (float) M_PI + 1.0f;
                    out[1](x, y, z) = 0.0f;
                    out[2](x, y, z) = phi * 4.0f / (float) M_PI;
                } else if (M_PI / 4.0f < phi && phi <= M_PI / 2.0f) {
                    out[0](x, y, z) = 0.0f;
                    out[1](x, y, z) = phi * 4.0f / (float) M_PI - 1.0f;
                    out[2](x, y, z) = 1.0f;
                } else if (M_PI / 2.0f < phi && phi <= 3 * M_PI / 4.0f) {
                    out[0](x, y, z) = 0.0f;
                    out[1](x, y, z) = 1.0f;
                    out[2](x, y, z) = -(phi - 3.0f * (float) M_PI / 4.0f) * 4.0f / (float) M_PI;
                } else if (3 * M_PI / 4.0f < phi && phi <= M_PI) {
                    out[0](x, y, z) = (phi - 3.0f * (float) M_PI / 4.0f) * 4.0f / (float) M_PI;
                    out[1](x, y, z) = -(phi - (float) M_PI) * 4.0f / (float) M_PI;
                    out[2](x, y, z) = 0.0f;
                }

                // out[0](x, y, z) = (180.0f / M_PI) * (phi / 2.0f);
                out[0](x, y, z) = a * out[0](x, y, z);
                out[1](x, y, z) = a * out[1](x, y, z);
                out[2](x, y, z) = a * out[2](x, y, z);

            }
        }
    }
}