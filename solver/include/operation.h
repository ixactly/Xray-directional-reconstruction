//
// Created by tomokimori on 23/08/31.
//

#ifndef SPMV_CSR_EXAMPLE_OPERATION_H
#define SPMV_CSR_EXAMPLE_OPERATION_H

#include "spMat.h"

void spmv(float alpha, csrSpMat& matA, DnVec& vecX, float beta, DnVec* vecY, cusparseHandle_t handle);
// sparseMat * sparseMat -> cusparseSpGEMM()
void spgemm();
#endif //SPMV_CSR_EXAMPLE_OPERATION_H
