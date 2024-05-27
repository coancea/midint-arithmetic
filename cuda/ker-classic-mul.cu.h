#ifndef KERNEL_CLASSIC_MUL
#define KERNEL_CLASSIC_MUL

#include "ker-helpers.cu.h"
#include "ker-addition.cu.h"

/**********************************************/
/*** Helpers for big-integer multiplication ***/
/**********************************************/

template<class S, class D>
__device__ inline
void computeIter64( uint32_t i, uint32_t j
                  , S* Ash, S* Bsh
                  , D& accum, uint32_t& carry
) {
    const uint32_t SHFT = 8*sizeof(S);
    S ai = Ash[i];
    S bj = Bsh[j];
    D ck = ((D)ai) * ((D)bj);

    S accum_prev = (S) (accum>>SHFT);
    accum += ck;
    carry += (  ((S)(accum>>SHFT)) < accum_prev );
    //if (accum < ck) carry++;
}

#if 0

/***************************************/
/*** Computing 4 elements per thread ***/
/***************************************/

template<class S, class D, uint32_t M>
__device__ inline
void combine2( D lh0, uint32_t c2
             , D lh1, uint32_t c3
             , S& l0_r, S& l1_r
             , S& h2_r, S& c3_r
             ) {
    S h1, l1, h2;
    uint32_t SHFT = 8 * sizeof(S);
    l0_r = (S) lh0;
    h1   = (S) (lh0 >> SHFT);
    l1   = (S) lh1;
    h2   = (S) (lh1 >> SHFT);
    
    l1_r = l1 + h1;
    h2_r = h2 + ( c2 + (l1_r < l1) );
    c3_r = c3 + (h2_r < h2);
}

template<class S, class D, uint32_t M>
__device__ inline
void convolution4( uint32_t k1, S* Ash, S* Bsh 
                 , S& l00, S& l01, S& h02, S& c03
) {    
    D accum_0 = 0, accum_1 = 0;
    uint32_t carry_0 = 0, carry_1 = 0;
    
    { 
        for(int kk = 0; kk <= k1; kk++) {
            uint32_t i = kk;
            uint32_t j = k1 - i;
            computeIter64<S,D>(i, j,   Ash, Bsh, accum_0, carry_0);
            computeIter64<S,D>(i, j+1, Ash, Bsh, accum_1, carry_1);
        }
        computeIter64<S,D>(k1+1, 0, Ash, Bsh, accum_1, carry_1);
        combine2<S,D,M>(accum_0, carry_0, accum_1, carry_1, l00, l01, h02, c03);
    }
}

template<class S, class D, uint32_t M>
__device__ inline
void wrapperConv4( S* Ash0, S* Bsh0 
                 , S& l00, S& l01, S& h02, S& c03
                 , S& l10, S& l11, S& h12, S& c13
) {
    const uint32_t offset = ( threadIdx.x / (M/4) ) * M;
    S* Ash = Ash0 + offset;
    S* Bsh = Bsh0 + offset;
        
    uint32_t ltid = threadIdx.x % (M/4);
    { // first half
        uint32_t k1 = 2*ltid;
        convolution4<S,D,M>( k1, Ash, Bsh, l00, l01, h02, c03 );
    }


    { // second half
        uint32_t k2 = M - 2*ltid - 2;
        convolution4<S,D,M>( k2, Ash, Bsh, l10, l11, h12, c13 );
    }
}

template<class S, uint32_t n> // e.g., <uint32_t, uint64_t, 32>
__device__ inline 
void from4Reg2Shm( S l00, S l01, S h02, S c03
                 , S l10, S l11, S h12, S c13
                 , S* Lsh, S* Hsh
) {
    const uint32_t offset = ( threadIdx.x / (n/4) ) * n;
    uint32_t tid_mod_m = threadIdx.x % (n/4);
    
    int32_t twoltid = offset + 2*tid_mod_m;
    {
        Lsh[twoltid]   = l00;
        Lsh[twoltid+1] = l01;
        Hsh[twoltid+2] = h02;
        Hsh[twoltid+3] = c03;
    }
    __syncthreads();
    int32_t n_m_2ltid = offset + n - 2*tid_mod_m;
    {
        Lsh[n_m_2ltid - 2] = l10;
        Lsh[n_m_2ltid - 1] = l11;
        
        S high  = h12; 
        S carry = c13;
        uint32_t ind   = n_m_2ltid; 
        if( tid_mod_m == 0 ) {
            high  = 0;
            carry = 0;
            ind   = offset;
        }
        Hsh[ind]   = high;
        Hsh[ind+1] = carry;
    }
}

/**************************************************/
/*** Main function for classical multiplication ***/
/**************************************************/

template<class Base, uint32_t IPB, uint32_t M>
__device__ 
void bmulRegs ( typename Base::uint_t* Ash
              , typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[4]
              , typename Base::uint_t Brg[4]
              , typename Base::uint_t Rrg[4]
              ) 
{
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    const uint32_t Q = 4;
    
    // 1. copy from global to shared to register memory
    cpReg2Shm<uint_t,Q>( Arg, Ash );
    cpReg2Shm<uint_t,Q>( Brg, Bsh );
    __syncthreads();
  
    // 2. perform the convolution
    uint_t l00, l01, h02, c03, l10, l11, h12, c13;
    wrapperConv4<uint_t, ubig_t, M>
        ( Ash, Bsh, l00, l01, h02, c03, l10, l11, h12, c13 );
    __syncthreads();

    typename Base::uint_t* Lsh = Ash;
    typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    from4Reg2Shm<uint_t, M>
            ( l00, l01, h02, c03
            , l10, l11, h12, c13
            , Lsh, Hsh
            );
    __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[4];
    cpShm2Reg<uint_t,Q>( Lsh, Lrg );
    uint_t Hrg[4];
    cpShm2Reg<uint_t,Q>( Hsh, Hrg );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, M, Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg );
}

/*********************************************/
/*** One (Classical) Multiplication Kernel ***/
/*********************************************/

template<typename Base, uint32_t IPB, uint32_t M>
__global__ void bmulKer ( typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        )
{
    using uint_t = typename Base::uint_t;
    const uint32_t Q = 4; // built-in sequentialization factor 4
    const uint32_t shmem_len = IPB*M;
#if 0  
    extern __shared__ char sh_mem_char_cmul[];
    uint_t* Ash = (uint_t*) sh_mem_char_cmul;
    uint_t* Bsh = Ash + shmem_len;
#endif

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];

    uint_t Arg[Q];  
    cpGlb2Reg<uint_t,IPB,M,Q>(IPB,Ash, ass, Arg);
    uint_t Brg[Q];
    cpGlb2Reg<uint_t,IPB,M,Q>(IPB,Bsh, bss, Brg);
    __syncthreads();

    uint_t Rrg[Q];
    bmulRegs<Base, IPB, M>(Ash, Bsh, Arg, Brg, Rrg);
    
    cpReg2Glb<uint_t,IPB,M,Q>(IPB,Ash, Rrg, rss);
}

/*********************************************/
/*** A Kernel that computes the polynomial ***/
/***     (a^2 + b) * (b^2 + b) + a*b       ***/
/*** using 4 multiplications & 2 additions ***/
/*********************************************/

template<typename Base, uint32_t IPB, uint32_t M>
__global__ void polyKer ( typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        ) 
{
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    
    const uint32_t Q = 4;
    const uint32_t shmem_len = IPB*M;
    //const uint32_t n = Q*M;

#if 0  
    extern __shared__ char sh_mem_char_poly[];
    uint_t* Ash = (uint_t*) sh_mem_char_poly;
    uint_t* Bsh = Ash + shmem_len;
#endif

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)Ash;

    uint_t Arg[Q];  
    cpGlb2Reg<uint_t,IPB,M,Q>(IPB,Ash, ass, Arg);
    uint_t Brg[Q];
    cpGlb2Reg<uint_t,IPB,M,Q>(IPB,Bsh, bss, Brg);
    __syncthreads();

    // t1 = a*a = a^2
    uint_t t1[Q]; 
    bmulRegs<Base, IPB, M>(Ash, Bsh, Arg, Arg, t1);
    
    // t2 = t1 + b = a^2 + b
    uint_t t2[Q]; 
    baddRegs<uint_t,uint_t,carry_t,M,Q,Base::HIGHEST>(carry_shm, t1, Brg, t2);

    // t3 = b^2
    uint_t t3[Q];
    bmulRegs<Base, IPB, M>(Ash, Bsh, Brg, Brg, t3);

    // t4 = t3 + b = b^2 + b
    uint_t t4[Q];
    baddRegs<uint_t,uint_t,carry_t,M,Q,Base::HIGHEST>(carry_shm, t3, Brg, t4);

    // t5 = t2 * t4 = (a^2 + b) * (b^2 + b)
    uint_t t5[Q];
    bmulRegs<Base, IPB, M>(Ash, Bsh, t2, t4, t5);

    // t6 = a*b
    uint_t t6[Q];
    bmulRegs<Base, IPB, M>(Ash, Bsh, Arg, Brg, t6);

    // R = t5 + t6 = (a^2 + b) * (b^2 + b) + a*b
    uint_t Rrg[Q];
    baddRegs<uint_t,uint_t,carry_t,M,Q,Base::HIGHEST>(carry_shm, t5, t6, Rrg);
    
    cpReg2Glb<uint_t,IPB,M,Q>(Ash, Rrg, rss);

#if 0
        uint32_t ind = 2*threadIdx.x;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = ind + 1;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = M - 2*threadIdx.x - 2;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = ind + 1;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];        
#endif
}

#endif // end of old version in which q was hardwired to 4

/**********************************************/
/***   Arbitrary Sequentialization Factor   ***/
/**********************************************/

template<class S, uint32_t n, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ( S lhcs[2][Q+2], S* Lsh, S* Hsh ) {

#if 0
    for(int q=0; q<2*Q; q++) {
        Lsh[threadIdx.x*2*Q + q] = 0;
        Hsh[threadIdx.x*2*Q + q] = 0;
    }
#endif
    //__syncthreads();

    const uint32_t Q2 = 2*Q;
    const uint32_t offset = ( threadIdx.x / (n/Q2) ) * n;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);

    {    
        int32_t twoltid = offset + Q*tid_mod_m;
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[twoltid+q] = lhcs[0][q];
        }
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[twoltid+q] = 0;
        }
        Hsh[twoltid+Q]   = lhcs[0][Q];
        Hsh[twoltid+Q+1] = lhcs[0][Q+1];
    }
    //__syncthreads();
    {
        int32_t n_m_2ltid = offset + n - Q*tid_mod_m - Q;
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[n_m_2ltid + q] = lhcs[1][q];
        }
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[n_m_2ltid + q] = 0;
        }
        S high = lhcs[1][Q];
        S carry= lhcs[1][Q+1];
        uint32_t ind = n_m_2ltid + Q;
        if( tid_mod_m == 0 ) {
            high  = 0;
            carry = 0;
            ind   = offset;
        }
        Hsh[ind]   = high;
        Hsh[ind+1] = carry;
    }
}


template<class S, class D, uint32_t M, uint32_t Q>
__device__ inline
void combineQ( D accums[Q], uint32_t carrys[Q], S lhcs[Q+2] ) {
    uint32_t SHFT = 8 * sizeof(S);
    
    lhcs[0] = (S) accums[0];
    S h_res = (S) (accums[0] >> SHFT);
    S c_res = carrys[0];

    #pragma unroll
    for(int q=1; q<Q; q++) {
        S l = (S) accums[q];
        S h = (S) (accums[q] >> SHFT);
        lhcs[q] = l + h_res;
        h_res = h + (c_res + (lhcs[q] < l));
        c_res = carrys[q] + (h_res < h);
    }
    lhcs[Q]   = h_res;
    lhcs[Q+1] = c_res;
}

template<class S, class D, uint32_t M, uint32_t Q>
__device__ inline
void convolutionQ( uint32_t k1, S* Ash, S* Bsh, S lhcs[Q+2] ) {
    D        accums[Q]; 
    uint32_t carrys[Q];
    
    #pragma unroll
    for(int q=0; q<Q; q++) { 
        accums[q] = 0; 
        carrys[q] = 0; 
    }
    
    for(int kk = 0; kk <= k1; kk++) {
        uint32_t i = kk;
        uint32_t j = k1 - i;
            
        #pragma unroll
        for(int q=0; q<Q; q++) {
            computeIter64<S,D>( i, j+q,   Ash, Bsh, accums[q], carrys[q] );
        }
    }
        
    #pragma unroll
    for(int q=1; q<Q; q++) {
        #pragma unroll
        for(int i=0; i<Q-q; i++) {
            computeIter64<S,D>(k1+q, i, Ash, Bsh, accums[i+q], carrys[i+q]);
        }
    }
    combineQ<S,D,M,Q>(accums, carrys, lhcs);
}

template<class S, class D, uint32_t M, uint32_t Q>
__device__ inline
void wrapperConvQ( S* Ash0, S* Bsh0, S lhcs[2][Q+2] ) {
    const uint32_t offset = ( threadIdx.x / (M/(2*Q)) ) * M;
    S* Ash = Ash0 + offset;
    S* Bsh = Bsh0 + offset;
    
    uint32_t ltid = threadIdx.x % (M/(2*Q));
    { // first half
        uint32_t k1 = Q*ltid;
        convolutionQ<S,D,M,Q>(k1, Ash, Bsh, lhcs[0]);
    }

    { // second half
        uint32_t k2 = M - Q*ltid - Q;
        convolutionQ<S,D,M,Q>(k2, Ash, Bsh, lhcs[1]);
    }
}

template<class Base, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ 
void bmulRegsQ( typename Base::uint_t* Ash
              , typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[2*Q]
              , typename Base::uint_t Brg[2*Q]
              , typename Base::uint_t Rrg[2*Q]
              ) 
{
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    
    // 1. copy from global to shared to register memory
    cpReg2Shm<uint_t,2*Q>( Arg, Ash );
    cpReg2Shm<uint_t,2*Q>( Brg, Bsh );
    __syncthreads();
  
    // 2. perform the convolution
    uint_t lhcs[2][Q+2];
    wrapperConvQ<uint_t, ubig_t, M, Q>( Ash, Bsh, lhcs );
    __syncthreads();

    typename Base::uint_t* Lsh = Ash;
    typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    from4Reg2ShmQ<uint_t, M, Q>( lhcs, Lsh, Hsh );
    __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    cpShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    uint_t Hrg[2*Q];
    cpShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, M, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg );
}

/**
 * Assumption: Q evenly divides M
 */
template<typename Base, uint32_t IPB, uint32_t M, uint32_t Q>
__global__ void bmulKerQ( uint32_t num_instances
                        , typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        )
{
    using uint_t = typename Base::uint_t;
    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];

    uint_t Arg[Q];
    uint_t Brg[Q];
    { // read from global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Ash, ass, Arg);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Bsh, bss, Brg);
    }
    __syncthreads();

    uint_t Rrg[Q];
    bmulRegsQ<Base, IPB, M_lft, Q/2>(Ash, Bsh, Arg, Brg, Rrg);

    { // write to global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpReg2Glb<uint_t,IPB,M,Q>(ipb, Ash, Rrg, rss);
    }
}

template<typename Base, uint32_t IPB, uint32_t M, uint32_t Q>
__global__ void polyKerQ( uint32_t num_instances
                        , typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        ) 
{
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;

    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;
    
#if 0  
    extern __shared__ char sh_mem_char_poly[];
    uint_t* Ash = (uint_t*) sh_mem_char_poly;
    uint_t* Bsh = Ash + shmem_len;
#endif

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)Ash;
    
    uint_t Arg[Q];
    uint_t Brg[Q];
    { // read from global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Ash, ass, Arg);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Bsh, bss, Brg);
    }
    __syncthreads();

    // t1 = a*a = a^2
    uint_t t1[Q]; 
    bmulRegsQ<Base, IPB, M_lft, Q/2>(Ash, Bsh, Arg, Arg, t1);
    
    // t2 = t1 + b = a^2 + b
    uint_t t2[Q]; 
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>(carry_shm, t1, Brg, t2);

    // t3 = b^2
    uint_t t3[Q];
    bmulRegsQ<Base, IPB, M_lft, Q/2>(Ash, Bsh, Brg, Brg, t3);

    // t4 = t3 + b = b^2 + b
    uint_t t4[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>(carry_shm, t3, Brg, t4);

    // t5 = t2 * t4 = (a^2 + b) * (b^2 + b)
    uint_t t5[Q];
    bmulRegsQ<Base, IPB, M_lft, Q/2>(Ash, Bsh, t2, t4, t5);

    // t6 = a*b
    uint_t t6[Q];
    bmulRegsQ<Base, IPB, M_lft, Q/2>(Ash, Bsh, Arg, Brg, t6);

    // R = t5 + t6 = (a^2 + b) * (b^2 + b) + a*b
    uint_t Rrg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>(carry_shm, t5, t6, Rrg);
    
    { // write back to global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpReg2Glb<uint_t,IPB,M,Q>(ipb, Ash, Rrg, rss);
    }
#if 0
        uint32_t ind = 2*threadIdx.x;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = ind + 1;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = M - 2*threadIdx.x - 2;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];
        ind = ind + 1;
        rss[blockIdx.x*M + ind] += Lsh[ind] + Hsh[ind] + Tsh[ind] + Csh[ind];        
#endif
}


#endif // KERNEL_CLASSIC_MUL
