//
// Created by tomokimori on 23/02/17.
//

#include "moire.cuh"
#include "Volume.h"
#include "Params.h"
#include "cmath"

float colorIntensity(float phi, float shift) {
    float phi_redef = phi - shift;
    if (phi_redef > M_PI / 2.0f) {
        phi_redef = phi_redef - (float) M_PI;
    } else if (phi_redef < -M_PI / 2.0f) {
        phi_redef = phi_redef + (float) M_PI;
    }

    if (-M_PI / 2.0f < phi_redef && phi_redef <= -M_PI / 3.0f) {
        return 0.0f;
    } else if (-M_PI / 3.0f < phi_redef && phi_redef <= -M_PI / 6.0f) {
        return (float) (6.0f / M_PI) * (phi_redef + M_PI / 3.0f);
    } else if (-M_PI / 6.0f < phi_redef && phi_redef <= M_PI / 6.0f) {
        return (float) 1.0f;
    } else if (M_PI / 6.0f < phi_redef && phi_redef <= M_PI / 3.0f) {
        return (float) (-6.0f / M_PI) * (phi_redef - M_PI / 3.0f);
    } else if (M_PI / 3.0f < phi_redef && phi_redef <= M_PI / 2.0f) {
        return 0.0f;
    }
}

void calcSinFittingLimited(const Volume<float> ct[4], Volume<float> out[3], int size_x, int size_y, int size_z) {
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                const float I[4] = {ct[0](x, y, z), ct[1](x, y, z), ct[2](x, y, z), ct[3](x, y, z)};
                const float a = (I[0] + I[1] + I[2] + I[3]) / 4.0f;
                const float b = std::sqrt((I[0] - I[2]) * (I[0] - I[2]) + (I[1] - I[3]) * (I[1] - I[3])) / 4.0f;
                float phi = std::atan2(I[1] - I[3], I[0] - I[2]) / 2.0f;
                // float phi = std::atan2(-(y - size_y / 2.0f), x - size_x / 2.0f);

                // phi = (phi >= 0.0f) ? phi : (float) M_PI + phi;
                float r = std::sqrt(
                        (x - size_x / 2.0f) * (x - size_x / 2.0f) + (y - size_y / 2.0f) * (y - size_y / 2.0f));
                // std::cout << "(x, y, z): " << x << ", " << y << ", " << z << std::endl;
                // std::cout << I[0] << ", " << I[1] << ", " << I[2] << ", " << I[3] << ", " << I[4] << std::endl;
                // std::cout << ct[3](x, y, z) << std::endl;
                // r, g, b


                out[0](x, y, z) = colorIntensity(phi, 0.0f);
                // out[0](x, y, z) = a * out[0](x, y, z);
                out[1](x, y, z) = colorIntensity(phi, M_PI / 3.0f);
                out[2](x, y, z) = colorIntensity(phi, 2.0f * M_PI / 3.0f);

            }
        }
    }
}

void calcPseudoCT(Volume<float> *dst, const Volume<float> *ct, int size_x, int size_y, int size_z) {
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                const float I[4] = {ct[0](x, y, z), ct[1](x, y, z), ct[2](x, y, z), ct[3](x, y, z)};
                const float a = (I[0] + I[1] + I[2] + I[3]) / 4.0f;
                const float b = std::sqrt((I[0] - I[2]) * (I[0] - I[2]) + (I[1] - I[3]) * (I[1] - I[3])) / 4.0f;
                float phi = std::atan2(I[1] - I[3], I[0] - I[2]) / 2.0f;

                float r = std::sqrt(
                        (x - size_x / 2.0f) * (x - size_x / 2.0f) + (y - size_y / 2.0f) * (y - size_y / 2.0f));
                // std::cout << "(x, y, z): " << x << ", " << y << ", " << z << std::endl;
                // std::cout << I[0] << ", " << I[1] << ", " << I[2] << ", " << I[3] << ", " << I[4] << std::endl;
                // std::cout << ct[3](x, y, z) << std::endl;
                // r, g, b

                dst[0](x, z, y) = colorIntensity(phi, 0.0f);
                // dst[0](x, z, y) = a * ct[0](x, y, z);
                dst[1](x, z, y) = colorIntensity(phi, M_PI / 3.0f);
                dst[2](x, z, y) = colorIntensity(phi, 2.0f * M_PI / 3.0f);
                dst[3](x, z, y) = a;
                dst[4](x, z, y) = b;
                dst[5](x, z, y) = phi;

                /*
                dst[0](x, z, y) = ct[0](x, y, z);
                dst[1](x, z, y) = ct[1](x, y, z);
                dst[2](x, z, y) = ct[2](x, y, z);
                dst[3](x, z, y) = ct[3](x, y, z);
                 */


            }
        }
    }
}

void phi2color(Volume<float> *dst, const Volume<float>& angle, int size_x, int size_y, int size_z) {
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                // float phi = std::atan2(-(y - size_y / 2.0f), x - size_x / 2.0f);
                float phi = angle(x, y, z);
                // r, g, b

                dst[0](x, y, z) = colorIntensity(phi, 0.0f);
                dst[1](x, y, z) = colorIntensity(phi, M_PI / 3.0f);
                dst[2](x, y, z) = colorIntensity(phi, 2.0f * M_PI / 3.0f);
            }
        }
    }
}