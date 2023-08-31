//
// Created by tomokimori on 23/08/29.
//

#ifndef ADMM_HOSTCSRMAT_H
#define ADMM_HOSTCSRMAT_H

#include <vector>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdlib.h>
#include <stdio.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

class DnVec {
public :
    DnVec(int rows, float *h_values) : rows(rows) {
        cudaMalloc((void **) &d_values, rows * sizeof(float));
        cudaMemcpy(d_values, h_values, rows * sizeof(float), cudaMemcpyHostToDevice);

        cusparseCreateDnVec(&vec, rows, d_values, CUDA_R_32F);
    }

    ~DnVec() {
        cudaFree(d_values);
        cusparseDestroyDnVec(vec);
    }

    cusparseDnVecDescr_t* get() {
        return &vec;
    }

    void toHost(float* h_values) {
        cudaMemcpy(h_values, d_values, rows * sizeof(float), cudaMemcpyDeviceToHost);
    }

private :
    const int rows;
    float *d_values;
    cusparseDnVecDescr_t vec;
};


class csrSpMat {
public :
    csrSpMat(int rows, int cols, int nnz, int *h_offsets, int *h_columns, float *h_values) :
            rows(rows), cols(cols), nnz(nnz) {
        cudaMalloc((void **) &d_offsets, (rows + 1) * sizeof(int));
        cudaMalloc((void **) &d_columns, nnz * sizeof(int));
        cudaMalloc((void **) &d_values, nnz * sizeof(float));

        cudaMemcpy(d_offsets, h_offsets, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice);

        cusparseCreateCsr(&mat, rows, cols, nnz,
                          d_offsets, d_columns, d_values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    }

    ~csrSpMat() {
        cudaFree(d_offsets);
        cudaFree(d_columns);
        cudaFree(d_values);
        cusparseDestroySpMat(mat);
    }

    cusparseSpMatDescr_t* get() {
        return &mat;
    }

    static void createHandle() {
        cusparseCreate(&handle);
    }
    // toDense, toSparse

    /* This is a pointer type to an opaque cuSPARSE context,
     * which the user must initialize by calling prior to calling
     * cusparseCreate() any other library function.
     * The handle created and returned by cusparseCreate() must be
     * passed to every cuSPARSE function.
     */
    static cusparseHandle_t handle;

private :
    const int rows;
    const int cols;
    const int nnz;

    int *d_offsets, *d_columns;
    float *d_values;

    cusparseSpMatDescr_t mat;

};
#endif //ADMM_HOSTCSRMAT_H
