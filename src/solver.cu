__global__
void
initGridKernel(float* Pd, const float* P0d, unsigned nPts)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < nPts)
    {
        const float* pt0 = &P0d[n*3];
        float* pt = &Pd[n*3];

        pt[0] = pt0[0];
        pt[1] = pt0[1];
        pt[2] = pt0[2];
        
        n += blockDim.x * gridDim.x;
    }
}

__host__
void
initGrid(float* Pd, const float* P0d, unsigned nPts)
{
    const unsigned nMaxThreadsPerBlock = 512;
    
    unsigned nWholeBlocks = nPts / nMaxThreadsPerBlock;
    unsigned nResidualThreads = nPts - (nWholeBlocks * nMaxThreadsPerBlock);
    
    dim3 dimBlock(nWholeBlocks >= 1 ? nMaxThreadsPerBlock : nResidualThreads);
    dim3 dimGrid(nWholeBlocks + (nResidualThreads > 0 ? 1 : 0) ) ;
    
    initGridKernel<<<dimGrid, dimBlock>>>(Pd, P0d, nPts);
}