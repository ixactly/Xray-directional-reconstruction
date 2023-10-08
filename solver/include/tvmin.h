//
// Created by tomokimori on 23/06/09.
//

#ifndef INC_3DRECONGPU_TVMIN_H
#define INC_3DRECONGPU_TVMIN_H
#include "volume.h"

void totalVariationMinimized(Volume<float> &vol, float rho, float lambda, int iter);
#endif //INC_3DRECONGPU_TVMIN_H
