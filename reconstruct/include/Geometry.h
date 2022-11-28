//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_GEOMETRY_H
#define INC_3DRECONGPU_GEOMETRY_H

class Geometry {

public:
    Geometry(float sdd, float sod, float detSize, int voxel, int detect, int nProj) :
            sdd(sdd), sod(sod), detSize(detSize), voxel(voxel), detect(detect), nProj(nProj) {
        voxSize = 0.5f * sod * detSize / sdd;
    }

    float sdd; // Object-Detector Distance
    float sod; // Source-Object Distance

    float voxSize; // voxel size
    float detSize; // detector size

    int voxel;
    int detect;
    int nProj;
};

class BasisVector {
public:
    BasisVector(int x, int y, int z) : x(x), y(y), z(z) {
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
    };

    void rotateBasis(const double rot[9]) {
        vec[0] = rot[0] * x + rot[1] * y + rot[2] * z;
        vec[1] = rot[3] * x + rot[4] * y + rot[5] * z;
        vec[2] = rot[6] * x + rot[7] * y + rot[8] * z;
    }

    const int *getVec() const {
        return vec;
    }

private:
    int x;
    int y;
    int z;
    int vec[3];
};

#endif //INC_3DRECONGPU_GEOMETRY_H
