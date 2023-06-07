//
// Created by tomokimori on 23/02/17.
//

#ifndef INC_3DRECONGPU_MOIRE_CUH
#define INC_3DRECONGPU_MOIRE_CUH

#include "Volume.h"
#include "Params.h"

void calcSinFittingLimited(const Volume<float> ct[4], Volume<float> out[3], int size_x, int size_y, int size_z);
void calcPseudoCT(Volume<float> *dst, const Volume<float> *ct, int size_x, int size_y, int size_z);
void phi2color(Volume<float> *dst, const Volume<float>& angle, int size_x, int size_y, int size_z);
void flipAxis(Volume<float> &dst, const Volume<float> &src, int size_x, int size_y, int size_z);
#endif //INC_3DRECONGPU_MOIRE_CUH
