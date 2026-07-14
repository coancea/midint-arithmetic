#ifndef RADIX_SORT_KERS
#define RADIX_SORT_KERS

struct RadixMeta32x8x23 {
    using uint = uint32_t;
    
    static const uint32_t largest = 0xFFFFFFFF;
    static const uint32_t bits = 32;
    static const uint32_t lgH  = 8;
    static const uint32_t Q    = 22;
    static const uint32_t B    = 256;
};

struct Test32x8x4 {
    using uint = uint32_t;
    
    static const uint32_t largest = 0xFFFFFFFF;
    static const uint32_t bits = 32;
    static const uint32_t lgH  = 5;
    static const uint32_t Q    = 7;
    static const uint32_t B    = 32;
};



template<class S, uint32_t B, uint32_t Q, S largest>
__device__ inline
void cpGlb2Reg ( uint64_t N, S* shmem, S* ass, S Arg[Q] ) {
    const uint32_t M = B * Q;

    // 1. read from global to shared memory
    const uint64_t glb_offs = blockIdx.x * M;
    
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*B + threadIdx.x;
        uint64_t glb_pos = glb_offs + loc_pos;

        S el = largest;
        if( glb_pos < N ) {
            el = ass[glb_pos];
        }
        shmem[loc_pos] = el;
    }
    __syncthreads();
    // 2. read from shmem to regs
    for(int i=0; i<Q; i++) {
        Arg[i] = shmem[Q*threadIdx.x + i];
    }
    __syncthreads();
}

template<typename uint, int num_bits> __device__ inline
int getBits(int bit_beg, uint x) {
    int mask = (1 << num_bits) - 1;
    int res  = (x >> bit_beg) & mask;
    return res;
}

template<typename uint> __device__ inline
int isBitUnset(int bit_num, uint x) {
    int shft = x >> bit_num;
    return 1 - (shft & 1);
}

template<typename Meta> __global__ void
//__launch_bounds__(1<<Meta::lgH, 1024/(1<<Meta::lgH))
ker1( size_t N, int bit_beg, typename Meta::uint* data_keys_in, uint16_t* d_histo ) {
    using uint = typename Meta::uint;

    const int histo_len = 1 << Meta::lgH;
    __shared__ uint32_t histo[histo_len];

    // initialize histogram
    for(int t=threadIdx.x; t<histo_len; t+=Meta::B) {
        histo[t] = 0;
    }
     __syncthreads();

    // compute histogram in shared memory
    {
        const uint64_t glb_offs = blockIdx.x * (Meta::B * Meta::Q);
        for(int i=0; i<Meta::Q; i++) {
            uint32_t loc_pos = i * Meta::B + threadIdx.x;
            uint64_t glb_pos = glb_offs + loc_pos;
            if(glb_pos < N) {
                uint el = data_keys_in[glb_pos];
                uint32_t  bin = getBits<uint, Meta::lgH>(bit_beg, el);
                atomicAdd(&histo[bin], 1);
            }
        }
    }
    __syncthreads();
    // write histogram to global memory
    {
        size_t glb_offs = blockIdx.x*histo_len;
        for(int t=threadIdx.x; t<histo_len; t+=Meta::B) {
            d_histo[glb_offs + t] = (uint16_t) histo[t];
        }
    }
}

// Assumkes Meta::B >= 2^{Meat::lgH}
template<typename Meta> __global__
__launch_bounds__(Meta::B, 1024/Meta::B)
void ker2( size_t N, int bit_beg
         , uint16_t* d_histo
         , uint64_t* d_histoST
         , typename Meta::uint* data_keys_in
         , typename Meta::uint* data_keys_out 
) {
    using uint = typename Meta::uint;

    //__shared__ uint elmshm[Meta::Q * Meta::B];
    // Meta::Q * Meta::B > 3*histo_len*sizeof(uint32_t)
    extern __shared__ uint64_t sh_mem_64[];
    uint* elmshm = (uint*) sh_mem_64;

    const int histo_len = 1 << Meta::lgH;
    uint64_t* histo_glb = (uint64_t*)sh_mem_64;
    uint16_t* histo_loc = (uint16_t*) (histo_glb+histo_len);
    
    uint     elms[Meta::Q];

    // copy the input elements from global to shared to registers
    cpGlb2Reg<uint, Meta::B, Meta::Q, Meta::largest>(N, elmshm, data_keys_in, elms);

    uint16_t thd_offset = Meta::Q*threadIdx.x;

    for(int i=0; i < Meta::lgH; i++) {

        // scan inclusive per thread
        uint16_t acc = 0;
        for(int q=0; q<Meta::Q; q++) {
            uint16_t zeroone = (uint16_t)isBitUnset<uint>(bit_beg + i, elms[q]);
            acc += zeroone;
            //tmp[q] = acc;
        }

        // publish last element of each thread to shared memory
        histo_loc[threadIdx.x] = acc;

        // scan across the threads in the block: ToDo Implement!
        uint16_t res = scanIncBlock< Add<uint16_t> >(histo_loc, threadIdx.x);
        __syncthreads();
        histo_loc[threadIdx.x] = res; 
        __syncthreads();

        // compute the split point and the prefix for each thread
        int16_t split = histo_loc[Meta::B-1];
        if (threadIdx.x == 0) acc = 0;
        else                  acc = histo_loc[threadIdx.x-1];
#if 0
        uint16_t tmp [Meta::Q];
        // apply the prefix for each thread. This concludes the scan
        for(int q=0; q<Meta::Q; q++) {
            //tmp[q] += acc;
            uint16_t zeroone = (uint16_t)isBitUnset<uint>(bit_beg + i, elms[q]);
            acc += zeroone;
            tmp[q] = acc;
        }
        __syncthreads();

        // permute in shared memory
        for(int q=0; q<Meta::Q; q++) {
            uint elm = elms[q];
            int pos;
            int zeroone = isBitUnset<uint>(bit_beg + i, elm);
            if(zeroone == 1) {
                pos = tmp[q] - 1;
            } else {
                pos = split + thd_offset + q - tmp[q];
            }
            elmshm[pos] = elm;
        }
#else
        // fused version that does not need tmp[Q]: this would be very useful,
        // when we also record the permutation throughout this loop, so we can
        // permute at the end the data associated with the keys.
        __syncthreads();
        for(int q=0; q<Meta::Q; q++) {
            uint elm = elms[q];
            uint16_t zeroone = (uint16_t)isBitUnset<uint>(bit_beg + i, elm);
            acc += zeroone;
            int pos;
            if(zeroone == 1) {
                pos = acc - 1;
            } else {
                pos = split + thd_offset + q - acc;
            }
            elmshm[pos] = elm;
        }
#endif
        __syncthreads();
        // load back to registers
        if(i < Meta::lgH-1) {
            for(int q=0; q<Meta::Q; q++) {
                elms[q] = elmshm[thd_offset+q];
            }
        } else {
            for(int q=0; q<Meta::Q; q++) {
                uint32_t loc_pos = q*Meta::B + threadIdx.x;
                elms[q] = elmshm[loc_pos];
            }
        }
        __syncthreads();

    } // end loop of count Meta:lgH

#if 1

    #if 1
    // copy histogram from global to shared memory
    {
        size_t glb_offs = blockIdx.x*histo_len;
        for(int t=threadIdx.x; t<histo_len; t+=Meta::B) {
            uint16_t loc_el = d_histo[glb_offs + t];            
            uint64_t glb_el = d_histoST[glb_offs + t];
            histo_glb[t] = glb_el - loc_el;
            histo_loc[t] = loc_el;
        }
    }
    __syncthreads();

    { // scan in-place histo_loc, then store it as scan exclusive
        int res = scanIncBlock< Add<uint16_t> >(histo_loc, threadIdx.x);
        if(threadIdx.x < histo_len-1)
            histo_glb[threadIdx.x+1] -= res;
        __syncthreads();
    }
    #else
    {
        size_t glb_offs = blockIdx.x*histo_len;
        for(int t=threadIdx.x; t<histo_len; t+=Meta::B) {         
            histo_glb[t] = d_histoST[glb_offs + t];
        }
        __syncthreads();
    }
    #endif

    // write the keys to global memory
    for(int q=0; q<Meta::Q; q++) {
        uint elm = elms[q];
        int  bin = getBits<uint, Meta::lgH>(bit_beg, elm);
        size_t  glb_offs = histo_glb[bin]; 
        // subtraction was performed above in scan in-place, i.e., histo_glb[bin] - histo_loc[bin];
        size_t  glb_pos  = glb_offs + (q*Meta::B + threadIdx.x);

        if(glb_pos < N)
            data_keys_out[glb_pos] = elm;
    }
#else
    {
        size_t glb_offs = blockIdx.x * (Meta::Q * Meta::B);
        for(int q=0; q<Meta::Q; q++) {
            uint32_t loc_pos = q*Meta::B + threadIdx.x;
            uint32_t glb_pos = glb_offs + loc_pos;// elm = elms[q];
            if(glb_pos < N)
                data_keys_out[glb_pos] = elms[q]; //elmshm[loc_pos];
        }
    }
#endif
}

template<typename Meta> __global__ void
kerHisto( uint16_t* d_histo, uint64_t* d_histoTST ) {

    const int histo_len = 1 << Meta::lgH;
    __shared__ uint16_t histo_loc[histo_len];

    uint64_t glb_offset = blockIdx.x*(1<<Meta::lgH);
    uint16_t loc_el = d_histo[ glb_offset + threadIdx.x ];

    uint16_t pos = threadIdx.x + 1;
    uint16_t val = loc_el;
    if(threadIdx.x == histo_len-1) { pos = 0; val = 0; }
    histo_loc[pos] = val;
    __syncthreads();

    int res = scanIncBlock< Add<uint16_t> >(histo_loc, threadIdx.x);
    d_histoTST[ glb_offset + threadIdx.x ] -= (loc_el + res);
}

#endif // RADIX_SORT_KERS