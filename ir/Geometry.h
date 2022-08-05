//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_GEOMETRY_H
#define INC_3DRECONGPU_GEOMETRY_H

class GeometryCUDA {

public:
    GeometryCUDA(double sdd, double sod, double detSize) : sdd(sdd), sod(sod), detSize(detSize) {
        voxSize = sod * detSize / sdd;
    }

    double sdd; // Object-Detector Distance
    double sod; // Source-Object Distance

    double voxSize; // voxel size
    double detSize; // detector size

};

#endif //INC_3DRECONGPU_GEOMETRY_H
