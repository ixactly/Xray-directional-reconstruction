//
// Created by tomokimori on 22/12/31.
//

#ifndef PCA_PCA_CUH
#define PCA_PCA_CUH

#include "../../reconstruct/include/Volume.h"
void calcEigenVector(const Volume<float> *ct, Volume<float> *md, int x, int y, int z);
#endif //PCA_PCA_CUH
