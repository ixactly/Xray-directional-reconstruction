//
// Created by tomokimori on 22/12/31.
//

#ifndef PCA_PCA_CUH
#define PCA_PCA_CUH

#include "../../reconstruct/include/Volume.h"
void calcEigenVector(const Volume<float> *ct, Volume<float> *md, Volume<float> *evalue, int x, int y, int z);
void calcPartsAngle(const Volume<float> md[3], Volume<float> angle[2], int x, int y, int z);
void rodriguesRotation(double n_x, double n_y, double z, double theta);
#endif //PCA_PCA_CUH
