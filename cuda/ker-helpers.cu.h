#ifndef KERNEL_HELPERS
#define KERNEL_HELPERS

#define WARP   (32)
#define lgWARP  (5)

#define HIGHEST32 ( 0xFFFFFFFF )
#define HIGHEST64 ( 0xFFFFFFFFFFFFFFFF )

typedef unsigned __int128 uint128_t;

struct U64bits {
    using uint_t = uint64_t;
    using sint_t = int64_t;
    using ubig_t = unsigned __int128;
    using carry_t= uint32_t;
    static const int32_t  bits = 64;
    static const uint_t HIGHEST = 0xFFFFFFFFFFFFFFFF;
};

struct U32bits {
    using uint_t = uint32_t;
    using sint_t = int32_t;
    using ubig_t = uint64_t;
    using carry_t= uint32_t;
    static const int32_t  bits = 32;
    static const uint_t HIGHEST = 0xFFFFFFFF;
};

#define LIFT_LEN(m,q) (((m + q - 1) / q) * q)

/***********************************************************/
/*** Remapping to/from Gobal, Shared and Register Memory ***/
/***********************************************************/

#if 0
template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2Sh ( S* ass, S* bss
              , S* Ash, S* Bsh 
) { 
    // 1. read from global to shared memory
    uint64_t glb_offs = blockIdx.x * (IPB * M);

    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M/Q) + threadIdx.x;
        S tmp_a = 0, tmp_b = 0;
        //if(loc_pos < IPB*M) 
        {
            tmp_a = ass[glb_offs + loc_pos];
            tmp_b = bss[glb_offs + loc_pos];
        }
        Ash[loc_pos] = tmp_a;
        Bsh[loc_pos] = tmp_b;
    }
}

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpSh2Glb(S* Hsh, S* rss) { 
    // 3. write from shared to global memory
    uint64_t glb_offs = blockIdx.x * (IPB * M);

    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M/Q) + threadIdx.x;
        //if(loc_pos < IPB*M) 
        {
            rss[glb_offs + loc_pos] = Hsh[loc_pos];
        }
    }
}

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2Reg ( volatile S* shmem, S* ass, S Arg[Q] ) { 
    // 1. read from global to shared memory
    uint64_t glb_offs = blockIdx.x * (IPB * M);

    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M/Q) + threadIdx.x;
        S tmp_a = 0;
        //if(loc_pos < IPB*M) 
        {
            tmp_a = ass[glb_offs + loc_pos];
        }
        shmem[loc_pos] = tmp_a;
    }
    __syncthreads();
    // 2. read from shmem to regs
    for(int i=0; i<Q; i++) {
        Arg[i] = shmem[Q*threadIdx.x + i];
    }
}

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpReg2Glb ( volatile S* shmem , S Rrg[Q], S* rss ) { 
    // 1. write from regs to shared memory
    for(int i=0; i<Q; i++) {
        shmem[Q*threadIdx.x + i] = Rrg[i];
    }
    __syncthreads();
    // 2. write from shmem to global
    uint64_t glb_offs = blockIdx.x * (IPB * M);
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M/Q) + threadIdx.x;
        //if(loc_pos < IPB*M) 
        {
            rss[glb_offs + loc_pos] = shmem[loc_pos];
        }
    }
}

#else

/*** BUG ***
 * These two should receive extra params denoting the total glb-mem length,
 * since IPB might not divide the number of instances
 */

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2Reg ( uint32_t ipb, volatile S* shmem, S* ass, S Arg[Q] ) {
    const uint32_t M_lft = LIFT_LEN(M, Q); 
    // 1. read from global to shared memory
    const uint64_t glb_offs = blockIdx.x * (IPB * M);
    
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos_sh = i*(IPB*M_lft/Q) + threadIdx.x;
        uint32_t r = loc_pos_sh % M_lft;

        uint32_t loc_pos_glb= (loc_pos_sh / M_lft) * M + r;
        S el = 0;
        if( (r < M) && (loc_pos_sh / M_lft < ipb) ) {
            el = ass[glb_offs + loc_pos_glb];
        }
        shmem[loc_pos_sh] = el;       
    }
    __syncthreads();
    // 2. read from shmem to regs
    for(int i=0; i<Q; i++) {
        Arg[i] = shmem[Q*threadIdx.x + i];
    }
}

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpReg2Glb ( uint32_t ipb, volatile S* shmem , S Rrg[Q], S* rss ) { 
    const uint32_t M_lft = LIFT_LEN(M, Q);
    
    // 1. write from regs to shared memory
    uint32_t ind_ipb = threadIdx.x / (M_lft/Q);
    for(int i=0; i<Q; i++) {
        uint32_t r = (Q*threadIdx.x + i) % M_lft;
        uint32_t loc_ind = ind_ipb*M + r;
        if(r < M)
            shmem[loc_ind] = Rrg[i];
    }

    __syncthreads();

    // 2. write from shmem to global
    const uint64_t glb_offs = blockIdx.x * (IPB * M);
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M_lft/Q) + threadIdx.x;
        if(loc_pos < ipb * M) {
            rss[glb_offs + loc_pos] = shmem[loc_pos];
        }
    }
}
#endif

template<class S, uint32_t Q>
__device__ inline
void cpReg2Shm ( S Rrg[Q], volatile S* shmem ) { 
    for(int i=0; i<Q; i++) {
        shmem[Q*threadIdx.x + i] = Rrg[i];
    }
}

template<class S, uint32_t Q>
__device__ inline
void cpShm2Reg ( volatile S* shmem, S Rrg[Q] ) { 
    for(int i=0; i<Q; i++) {
        Rrg[i] = shmem[Q*threadIdx.x + i];
    }
}

/******************************************************/
/*** kernels for measuring the latency of basic ops ***/
/******************************************************/

template<typename T>
__global__ void additionKer( uint32_t n, T* rss ) {
    T gid = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if(gid > n) return;
    T r = 2*n + gid;
    for(int32_t i=0; i<n; i++) {
        r = r + gid; //% n;
    }
    rss[gid-1] = r;
}

template<typename T>
__global__ void modulusKer( uint32_t n, T* rss ) {
    T gid = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if(gid > n) return;
    T r = 2*n + gid;
    for(int32_t i=0; i<n; i++) {
        r += r % gid; //% n;
    }
    rss[gid-1] = r;
}

template<typename T>
__global__ void multiplyKer( uint32_t n, T* rss ) {
    T gid = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if(gid > n) return;
    T r = gid+n;
    for(int32_t i=0; i<n; i++) {
        r = r * gid;
    }
    rss[gid-1] = r;
}

/****************************/
/*** Latency of BasicsOps ***/
/****************************/

template<class uint_t>  // m is the size of the big word in Base::uint_t units
void testBasicOps ( int n, int runs ) {    
    uint_t* d_as;
    size_t mem_size_nums = n * sizeof(uint_t);
    
    cudaMalloc((void**) &d_as, mem_size_nums);
    
    const size_t B = 256;
    
    // 1. timing instrumentation addition
    {
        additionKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    
    
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<runs; i++) {
            additionKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs;

        gpuAssert( cudaPeekAtLastError() );

        printf( "Base Addition runs in: %.2f us\n", (double)elapsed );        
    }
        
    // 2. timing instrumentation multiplication
    {
        multiplyKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    
    
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<runs; i++) {
            multiplyKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs;

        gpuAssert( cudaPeekAtLastError() );

        printf( "Base Multiply runs in: %.2f us\n", (double)elapsed );
    }
            
    // 3. timing instrumentation multiplication
    {
        modulusKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<runs; i++) {
            modulusKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs;

        gpuAssert( cudaPeekAtLastError() );

        printf( "Base Modulus runs in: %.2f us\n", (double)elapsed );
    }
    
    cudaFree(d_as);
}

#endif //KERNEL_HELPERS

