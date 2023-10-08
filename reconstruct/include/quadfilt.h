//
// Created by tomokimori on 23/09/01.
//

#ifndef INC_3DRECONGPU_QUADFILT_H
#define INC_3DRECONGPU_QUADFILT_H

#include "volume.h"

void quadlicFormFilterCPU(Volume<float> voxel[3], Volume<float> *coefficient, float lambda);
#endif //INC_3DRECONGPU_QUADFILT_H
