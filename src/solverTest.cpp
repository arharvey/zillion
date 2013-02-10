#include "solverTest.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "cudaUtils.h"
#include "cusparseUtils.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
void
solverTest(int cudaDevice)
{
    cusparseHandle_t handle = 0;
    
    std::cout << "Solver test" << std::endl;
    
    // Initialize CUSPARSE library
    
    cusparseCheckStatus( cusparseCreate(&handle) );
    
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    
    //float A[] = {3, 2, 2, 6}; // Row major format
    float h_A[] = {1, 2, 1}; // Row major format

    const int nnz = sizeof(h_A)/sizeof(h_A[0]);
    
    int h_ArowPtr[] = {0, 1, nnz+0 /*Zero-based indexing*/};
    int h_AcolInd[] = {0, 0, 1};
    
    const int m = sizeof(h_ArowPtr)/sizeof(h_ArowPtr[0])-1;

    float h_B[] = {2, -8};
    const int nnzv = sizeof(h_B)/sizeof(h_B[0]);
    
    float h_x[nnzv] = {0, 0};
    
    std::cout << "m = " << m << ", nnz = " << nnz << ", nnzv = " << nnzv << std::endl;
    
    cusparseMatDescr_t descrA = 0;
    cusparseCheckStatus( cusparseCreateMatDescr(&descrA) );
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    // Transfer matrix data to GPU
    
    float* d_A = NULL;
    int* d_ArowPtr = NULL;
    int* d_AcolInd = NULL;
    
    float* d_B = NULL;
    float* d_x = NULL;
    
    float* d_alpha = NULL;
    
    cudaCheckError( cudaMallocT(d_A, nnz) );
    cudaCheckError( cudaMallocT(d_ArowPtr, m+1) );
    cudaCheckError( cudaMallocT(d_AcolInd, nnz) );
    cudaCheckError( cudaMallocT(d_B, nnzv) );
    cudaCheckError( cudaMallocT(d_x, nnzv) );
    cudaCheckError( cudaMallocT(d_alpha, 1) );
    
    cudaCheckError( cudaMemcpyT(d_A, h_A, nnz, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpyT(d_ArowPtr, h_ArowPtr, m+1, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpyT(d_AcolInd, h_AcolInd, nnz, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpyT(d_B, h_B, nnzv, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpyT(d_x, h_x, nnzv, cudaMemcpyHostToDevice) );
    
    float alpha = 1.0;
    cudaCheckError( cudaMemcpyT(d_alpha, &alpha, 1, cudaMemcpyHostToDevice) );
    
    cusparseSolveAnalysisInfo_t info;
    cusparseCheckStatus( cusparseCreateSolveAnalysisInfo(&info) );
    
    cusparseStatus_t s;

    s = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                m, nnz, descrA, d_A, d_ArowPtr, d_AcolInd,
                                info);
    cusparseCheckStatus(s);
#if 1
    s = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                             m, d_alpha, descrA, d_A, d_ArowPtr, d_AcolInd, 
                             info, d_B, d_x);
    cusparseCheckStatus(s);
#endif
    cudaDeviceSynchronize();
    
    cudaCheckError( cudaMemcpyT(h_x, d_x, nnzv, cudaMemcpyDeviceToHost) );
    
    for(unsigned n = 0; n < m; n++)
        std::cout << h_x[n] << std::endl;
    
    cusparseCheckStatus( cusparseDestroySolveAnalysisInfo(info) );
    
    cusparseCheckStatus( cusparseDestroy(handle) );
    handle = 0;
}
       

// ---------------------------------------------------------------------------
    
} // END NAMESPACE ZILLION
