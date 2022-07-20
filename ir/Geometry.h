//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_GEOMETRY_H
#define INC_3DRECONGPU_GEOMETRY_H

class GeometryCUDA {

public:
    GeometryCUDA(float sdd, float sod, float detSize) : sdd(sdd), sod(sod), detSize(detSize) {
        voxSize = sod * detSize / sdd;
    }

    float sdd; // Object-Detector Distance
    float sod; // Source-Object Distance

    float voxSize; // voxel size
    float detSize; // detector size

};

#endif //INC_3DRECONGPU_GEOMETRY_H
