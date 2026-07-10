#ifndef FFT_HELPER

template<class uint>
__host__ 
void packBBits(uint* input, const size_t M, const uint32_t B, uint* output) {
    // assumes the output is already zeroed
    //uint32_t* output = (uint32_t*)calloc(M, sizeof(uint32_t));
    const int word_num_bits = sizeof(uint)*8;
    
    size_t    out_idx = 0;    // Index of the current element in the output array
    uint32_t  bit_pos = 0;    // Number of bits currently written to output[out_idx]
    
    // Mask to extract only the first B bits (LSBs) from the input elements
    uint32_t mask_B = (1U << B) - 1;

    for (size_t i = 0; i < M; ++i) {
        // Extract the B bits
        uint32_t bits_to_pack = input[i] & mask_B;
        uint32_t bits_left = B;

        while (bits_left > 0) {
            // Stop if we exceed the output array boundaries
            if (out_idx >= M) break;

            uint32_t space_available = word_num_bits - bit_pos;
            uint32_t bits_to_write = std::min(bits_left, space_available);

            // Isolate the exact chunk of bits we can write into the current slot
            uint32_t chunk_mask = (1U << bits_to_write) - 1;
            uint32_t chunk = bits_to_pack & chunk_mask;

            // Shift chunk to its correct position and combine with output
            output[out_idx] |= (chunk << bit_pos);

            // Update remaining bits and tracking positions
            bits_to_pack >>= bits_to_write;
            bits_left -= bits_to_write;
            bit_pos += bits_to_write;

            // Move to the next output element if the current one is completely filled
            if (bit_pos == word_num_bits) {
                bit_pos = 0;
                out_idx++;
            }
        }
    }
}

template<class uint, class uintd>
__host__
void evaluatePolynomial(uint* A, const size_t m, const uint32_t b, uint* R) {
    // R is assumed initialized with zero
    const int word_num_bits = sizeof(uint)*8;
    for (size_t i = 0; i < m; ++i) {
        uintd val = A[i];
        
        // Calculate the exact bit position where A[i] * 2^(b*i) starts
        size_t   word_shift = (b * i) / word_num_bits;
        uint32_t bit_shift  = (b * i) % word_num_bits;

        // Shift the 32-bit coefficient into a 64-bit accumulator
        uintd shifted_val = val << bit_shift;

        // Propagate the addition and handles carries across the big integer array
        uintd carry = shifted_val;
        size_t idx = word_shift;

        while (carry > 0 && idx < m) {
            if(idx >= m/2) {
                printf("Logical error: the compacted FFT result is larger than the precision!\n");
                exit(1);
            }
            carry += R[idx];
            R[idx] = static_cast<uint>(carry);
            carry >>= word_num_bits; // Extract the carry for the next word
            idx++;
        }
    }
}

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
