#ifndef FFT_HELPER

template<typename P, uint32_t Q> 
__device__ inline void
splitFftReg ( typename P::uint_t  Rreg[Q]
            , typename P::uhlf_t* shmhf
            , typename P::uhlf_t  Rlw[Q]
            , typename P::uhlf_t  Rhc[Q]
            ) {
    using uint_t = typename P::uint_t;
    using uhlf_t = typename P::uhlf_t;
    const uint32_t base = P::base;
    
    const uint_t pp = (( ((uint_t)1) << base) - 1);

    // iter q=0
    uint_t word0 = Rreg[0];
    Rlw[0]       = ((uhlf_t) (word0 & pp)) * 2;
    uint_t tmp0  = word0 >> base;
    uhlf_t high0 = ((uhlf_t) (tmp0  & pp)) * 2;
    uhlf_t crry0 = ((uhlf_t) (tmp0  >> base)) * 2;
    
    // iter q = 1
    uint_t word1 = Rreg[1];
    uhlf_t low1  = ((uhlf_t) (word1 & pp)) * 2;
    uint_t tmp1  = word1 >> base;
    uhlf_t high1 = ((uhlf_t) (tmp1  & pp)) * 2;
    uhlf_t crry1 = ((uhlf_t) (tmp1  >> base)) * 2;
    
    Rlw[1] = low1 + high0;
    uhlf_t c1 = (Rlw[1] < high0);
    uhlf_t acc_high = high1 + 2*c1 + crry0;
    uhlf_t c2 = (acc_high < high1);
    uhlf_t acc_crry = crry1 + 2*c2;  
    
    #pragma unroll
    for(int q=2; q<Q; q++) {
        uint_t word  = Rreg[q];
        uhlf_t low   = ((uhlf_t) (word & pp)) * 2;
        uint_t tmp   = word >> base;
        uhlf_t high  = ((uhlf_t) (tmp  & pp)) * 2;
        uhlf_t crry = ((uhlf_t) (tmp  >> base)) * 2;
        
        Rlw[q] = low + acc_high;
        uhlf_t c1 = (Rlw[q] < low);
        
        acc_high = high + 2*c1 + acc_crry;
        uhlf_t c2 = (acc_high < high);
        
        acc_crry = 2*c2 + crry;      
    }

    shmhf[2*threadIdx.x]   = acc_high;
    shmhf[2*threadIdx.x+1] = acc_crry;
    __syncthreads();
    {
        uhlf_t high = 0;
        uhlf_t crry = 0;
        if(threadIdx.x > 0) {
            high = shmhf[2*(threadIdx.x-1)];
            crry = shmhf[2*(threadIdx.x-1)+1];
        }
        Rhc[0] = high;
        Rhc[1] = crry;
        for(int q=2; q<Q; q++)
            Rhc[q] = 0;
    }
    __syncthreads();
}

template<typename P, uint32_t M, uint32_t Q, uint32_t not_already_mul_with_2> 
__device__ inline void
baddRegMul2Fft( typename P::uhlf_t* shmhalf
              , typename P::uhlf_t  Arg[Q]
              , typename P::uhlf_t  Brg[Q]
              , typename P::uhlf_t  Res[Q]
) {
    using uhlf_t = typename P::uhlf_t;
    using uint_t = typename P::uint_t;
    
    if(not_already_mul_with_2) {
        for(int q=0; q<Q; q++) {
            Arg[q] = Arg[q] << 1;
            Brg[q] = Brg[q] << 1;
        }
    }
    
    const uhlf_t HIGHEST = (( ((uint_t)1) << P::base) - 1) * 2; 
    baddRegs<uhlf_t,uhlf_t,uhlf_t,M,Q,HIGHEST>( shmhalf, Arg, Brg, Res );
    
    for(int q=0; q<Q; q++) {
        uhlf_t carry = Res[q] & 1;
        Res[q] = (Res[q] >> 1) + carry;
    }
}

#endif // FFT_HELPER
