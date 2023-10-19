//
// Created by tomokimori on 23/09/27.
//

#ifndef INC_3DRECONGPU_POISSON_CPU_H
#define INC_3DRECONGPU_POISSON_CPU_H
#include "volume.h"

void poissonImageEdit(Volume<float>& dst, const Volume<float>* src, int loop);

#endif //INC_3DRECONGPU_POISSON_CPU_H
