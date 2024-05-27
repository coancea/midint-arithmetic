#ifndef KERNEL_FFT_MUL
#define KERNEL_FFT_MUL

#include "ker-fft-help.cu.h"

template <typename T> T min2(T a, T b) { return a < b ? a : b; }

// Reverse of j considered as integer of nb bits.
__device__ uint32_t bitReverse(uint32_t j, uint32_t nb) {
        uint32_t r = 0;
        for ( ; nb--; j >>= 1)
            r = (r << 1) | (j & 1);
        return r;
}

template <typename uint_t>
bool is2pow(uint_t n) {
    bool seen1 = false;
    for ( ; n ; n >>= 1) {
        if (n & 1) {
            if (seen1) return false;
            seen1 = true;
        }
    }
    return seen1;
}

// ceil(log[2](n))   1->0, 2->1, 3..4->2, 5..8->3, etc
template <typename uint_t>
uint32_t ceilLg(uint_t n) {
    size_t lastSeen = 0, numSeen = 0;
    for (unsigned i = 0; n; i++, n >>= 1) {
        if (n & 1) { 
            lastSeen = i; 
            numSeen++; 
        }
    }
    if (numSeen > 1) lastSeen++;
    return lastSeen;
}

// template <typename rep_t, rep_t modulus, typename ubig_t, typename srep_t>

struct FftPrime64 {
    using uint_t = uint64_t;
    using sint_t = int64_t;
    using ubig_t = unsigned __int128;
    using uhlf_t = uint32_t;
    
    static const int32_t  bits = 64;
    
    static const uint32_t n = 57;
    static const uint32_t k = 29;
    static const uint_t   p = 4179340454199820289;
    static const uint_t   g = 21;
    static const size_t   base = 31; //1 << 31;
};

struct FftPrime32 {
    using uint_t = uint32_t;
    using sint_t = int32_t;
    using ubig_t = uint64_t;
    using uhlf_t = unsigned short;
    
    static const int32_t  bits = 32;
    
    static const uint32_t n = 30;
    static const uint32_t k = 3;
    static const uint_t   p = 3221225473;
    static const uint_t   g = 13;
    static const size_t   base = 15; //1 << 15;
};


//template <typename rep_t, rep_t modulus, typename ubig_t, typename srep_t>
template<typename P>
class zmod_t {
    using rep_t  = typename P::uint_t;
    using ubig_t = typename P::ubig_t;
    using srep_t = typename P::sint_t;
    static const rep_t modulus = P::p;

public:
    __host__ __device__ static rep_t norm(const rep_t v) {
        return (0 <= v && v < modulus) ? v : v % modulus;
    }

    __host__ __device__ static rep_t add (const rep_t x, const rep_t y) {
        ubig_t r = ((ubig_t) x) + ((ubig_t) y);
        if (r >= modulus) r -= modulus;
        return (rep_t)r;
    }

    __host__ __device__ static rep_t sub (const rep_t x, const rep_t y) {
        rep_t r = x;
        if (x < y) r += modulus;
        return (r - y);
    }

    __host__ __device__ static rep_t neg(const rep_t x) {
        rep_t r = 0;
        if (x != 0) { r = modulus - x; }
        return r;
    }
    /////////////////////////

    __host__ __device__ static rep_t mul(const rep_t x, const rep_t y) {
        ubig_t r = ((ubig_t) x) * ((ubig_t) y);
        return (r % modulus);
    }
    
    __host__ __device__ static rep_t div(const rep_t x, const rep_t y) {
        //return (*this) * y.inv();
        return mul( x, inv(y) );
    }
    
    __host__ __device__ static rep_t inv(const rep_t x) {
        rep_t  a = x,  b = modulus;
        srep_t s = 1,  t = 0;
        while (b != 0) {
            srep_t q = a/b, r = a % b;
            a = b; b = r; 
            r = s - q*t;
            s = t; t = r; 
        }
        if (s < 0) s += modulus;
        return s;
    }
    // pow uses zmod_t arithmetic.
    __host__ __device__ static rep_t pow(const rep_t x, const rep_t n0) {
        if (n0 == 0) return 1;
        rep_t a = x, b = 1;
        rep_t n = n0;
        for ( ; n > 1; n >>= 1) { 
            if (n & 1) b = mul(a,b); 
            a = mul(a,a);
        }
        return mul(a,b);
    }
};


template<typename uint_t, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2ShFFT ( uint_t* ass, uint_t* bss
                 , uint_t* Ash, uint_t* Bsh
) { 
    // 1. read from global to shared memory
    uint64_t glb_offs = blockIdx.x * (IPB * M);

    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M/Q) + threadIdx.x;
        uint_t tmp_a = 0, tmp_b = 0;
        //if(loc_pos < IPB*M) 
        {
            tmp_a = ass[glb_offs + loc_pos];
            tmp_b = bss[glb_offs + loc_pos];
        }
        Ash[loc_pos] = tmp_a;
        Bsh[loc_pos] = tmp_b;
    }
}

template<typename uint_t, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpSh2GlbFFT(uint_t* Hsh, uint_t* rss) { 
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

template <typename rep_t>
__device__ void permuteIP( uint32_t k, uint32_t t, rep_t* xss) {    
    uint32_t j = bitReverse(k, t);
    if(j > k) {
        rep_t x_k = xss[k];
        rep_t x_j = xss[j];
        xss[j] = x_k;
        xss[k] = x_j;
    }
    __syncthreads();
}

template<typename P>
__host__ typename P::uint_t getOmega(uint32_t n) {
    // Fp     g   = Fp(Prime::g).pow((uint_t) 1 << (Prime::n - ceilLg(n)));
    using uint_t = typename P::uint_t;
    uint32_t lgn = ceilLg<uint_t>(n);
    uint_t shft = P::n - lgn;
    uint_t e = 1 << shft;
    return zmod_t<P>::pow(P::g, e);
}

template<typename P>
__host__ void mkOmegas(uint32_t n, typename P::uint_t omega, typename P::uint_t* omegas) {
    using uint_t = typename P::uint_t;
    using PF = zmod_t<P>;
    
    uint_t acc = 1;
    for(int i=0; i<n; i++) {
        omegas[i] = acc;
        acc = PF::mul(acc, omega); 
    }
}

#if 0
template<typename P, uint32_t n>
__device__ void fft1( uint32_t lgn, typename P::uint_t* omegas, typename P::uint_t* xss ) {
    using uint_t = typename P::uint_t;
    using PF = zmod_t<P>;
    
    //uint32_t lgn = ceilLg<uint_t>(n);
    permuteIP<uint_t>( threadIdx.x, lgn, xss );
    permuteIP<uint_t>( threadIdx.x + n/2, lgn, xss );
    
    for(int32_t q = 1; q <= lgn; q++) {
        int32_t L   = 1 << q;
        int32_t Ld2 = 1 << (q-1);
        int32_t r   = n >> q;
        
        int32_t k = threadIdx.x >> (q-1);
        int32_t j = threadIdx.x & (Ld2 - 1);
        int32_t kLj = k*L + j;
        uint_t omega_pow= omegas[r*j];
        uint_t tau      = PF::mul( omega_pow , xss[kLj + Ld2] );
        uint_t x_kLj    = xss[kLj];
        xss[kLj]       = PF::add(x_kLj, tau);
        xss[kLj + Ld2] = PF::sub(x_kLj, tau);
        __syncthreads();
    }
}

template<typename P, uint32_t n>
__device__ void ifft1( uint32_t lgn, typename P::uint_t* omegas_inv, typename P::uint_t* xss ) {
    using uint_t = typename P::uint_t;
    using PF = zmod_t<P>;
    
    //uint_t omega = getOmega(n);
    //uint_t omega_inv = PF::inv(omega);
    uint_t n_inv = PF::inv(n);
     
    fft1<P,n>( lgn, omegas_inv, xss );

    uint32_t ind = threadIdx.x;
    xss[ind] = PF::mul(n_inv, xss[ind]);
    ind += (n>>1);
    xss[ind] = PF::mul(n_inv, xss[ind]);

    __syncthreads();    
}

template<typename P, uint32_t M>
__global__ void bmulFFT1( uint32_t clgm
                       , typename P::uint_t* omegas
                       , typename P::uint_t* omegas_inv
                       , typename P::uint_t* ass
                       , typename P::uint_t* bss
                       , typename P::uint_t* rss 
) {
    using pft    = zmod_t<P>;
    using uint_t = typename P::uint_t;
    //const uint32_t shmem_len = M;
    
    extern __shared__ char sh_mem[];
    uint_t* Ash = (uint_t*) sh_mem;
    uint_t* Bsh = Ash + M;
    uint_t* Tsh = Bsh + M;
//    __shared__ uint_t Ash[shmem_len];
//    __shared__ uint_t Bsh[shmem_len];
//    __shared__ uint_t Tsh[shmem_len];

    cpGlb2ShFFT<uint_t, 1, M, 2>(ass, bss, Ash, Bsh);

    __syncthreads();

    fft1<P,M>(clgm, omegas, Ash);
    fft1<P,M>(clgm, omegas, Bsh);
    
    //let xy1 = map2 (pf.*) (x'[0:n/2]) (y'[0:n/2])
    //let xy2 = map2 (pf.*) (x'[n/2:n]) (y'[n/2:n])
    uint32_t ind = threadIdx.x;
    Tsh[ind] = pft::mul(Ash[ind], Bsh[ind]);
    ind += (M >> 1);
    Tsh[ind] = pft::mul(Ash[ind], Bsh[ind]);
    
    __syncthreads();
    ifft1<P,M>(clgm, omegas_inv, Tsh);
    
    __syncthreads();
    cpSh2GlbFFT<uint_t, 1, M, 2>(Tsh, rss);
}
#endif
////////////////////////////////////////////////
/// New Version 
////////////////////////////////////////////////

template<typename P, uint32_t n, uint32_t Q>
__device__ void fft ( typename P::uint_t* xss
                    , uint32_t lgn
                    , typename P::uint_t* omegas
                    , typename P::uint_t Arg[2*Q]
                    , typename P::uint_t Rrg[2*Q]
                    ) 
{
    using uint_t = typename P::uint_t;
    using PF = zmod_t<P>;
    
    cpReg2Shm<uint_t,2*Q>( Arg, xss );
    __syncthreads();
    
    //uint32_t lgn = ceilLg<uint_t>(n);
    #pragma unroll
    for(int32_t v=0; v<2*Q; v++) {
        int32_t vtid = threadIdx.x + v*blockDim.x;
        permuteIP<uint_t>( vtid, lgn, xss );
    }
    
    for(int32_t q = 1; q <= lgn; q++) {
        int32_t L   = 1 << q;
        int32_t Ld2 = 1 << (q-1);
        int32_t r   = n >> q;
        
        #pragma unroll
        for(int32_t v=0; v<Q; v++) {
            int32_t vtid = threadIdx.x + v*blockDim.x;
            int32_t k = vtid >> (q-1);
            int32_t j = vtid & (Ld2 - 1);
            int32_t kLj = k*L + j;
            uint_t omega_pow= omegas[r*j];
            uint_t tau      = PF::mul( omega_pow , xss[kLj + Ld2] );
            uint_t x_kLj    = xss[kLj];
            xss[kLj]       = PF::add(x_kLj, tau);
            xss[kLj + Ld2] = PF::sub(x_kLj, tau);
        }
        __syncthreads();
    }
    
    cpShm2Reg<uint_t,2*Q>( xss, Rrg );
    __syncthreads();

}

template<typename P, uint32_t M, uint32_t Q>
__device__ void ifft( typename P::uint_t* xss
                    , uint32_t lgn
                    , typename P::uint_t  n_inv
                    , typename P::uint_t* omegas_inv
                    , typename P::uint_t Arg[2*Q]
                    , typename P::uint_t Rrg[2*Q] 
                    ) 
{
    using uint_t = typename P::uint_t;
    using PF = zmod_t<P>;
    
    fft<P,M,Q>( xss, lgn, omegas_inv, Arg, Rrg );

    //uint_t n_inv = PF::inv(M);
    for(int i=0; i<2*Q; i++) {
        Rrg[i] = PF::mul(n_inv, Rrg[i]);
    }
}

/**
 * ToDo: 
 *   1. add invM as parameter!
 *   2. find a way to pass halfs to fft.
 */
template<typename P, uint32_t M, uint32_t Q> __device__ 
inline void bmulFftReg( uint32_t clgm
                      , typename P::uint_t  invM
                      , typename P::uint_t* omegas
                      , typename P::uint_t* omegas_inv
                      , typename P::uhlf_t  Ahlf[2*Q]
                      , typename P::uhlf_t  Bhlf[2*Q]
                      , typename P::uhlf_t* shmhalf
                      , typename P::uhlf_t  Rhlf[2*Q]
) {
    using pft    = zmod_t<P>;
    using uint_t = typename P::uint_t;
    using uhlf_t = typename P::uhlf_t;
    uint_t* shmem   = (uint_t*)shmhalf;

    uint_t Arg[2*Q];
    for(int q=0;q<2*Q;q++) Arg[q] = Ahlf[q];
    
    uint_t Afft[2*Q];
    fft<P,M,Q>(shmem, clgm, omegas, Arg, Afft);
    
    uint_t Brg[2*Q];
    for(int q=0;q<2*Q;q++) Brg[q] = Bhlf[q];
    
    uint_t Bfft[2*Q];
    fft<P,M,Q>(shmem, clgm, omegas, Brg, Bfft);
    
    uint_t Trg[2*Q], Rrg[2*Q];
    for(int i=0; i<2*Q; i++) {
        Trg[i] = pft::mul(Afft[i], Bfft[i]);
    }

    ifft<P,M,Q>(shmem, clgm, invM, omegas_inv, Trg, Rrg);

    uhlf_t Rlw[2*Q];
    uhlf_t Rhc[2*Q];
    splitFftReg<P,2*Q>(Rrg, shmhalf, Rlw, Rhc);

#if 1
    baddRegMul2Fft<P, M, 2*Q, 0>( shmhalf, Rlw, Rhc, Rhlf );
#else    
    // add them with carry propagation of course
    const uhlf_t HIGHEST = (( ((uint_t)1) << P::base) - 1) * 2; 
    baddRegs<uhlf_t,uhlf_t,uhlf_t,M,2*Q,HIGHEST>( shmhalf, Rlw, Rhc, Rhlf );
    
    for(int q=0; q<2*Q; q++) {
        uhlf_t carry = Rhlf[q] & 1;
        Rhlf[q] = (Rhlf[q] >> 1) + carry;
    }
#endif
}

template<typename P, uint32_t M, uint32_t Q> __global__ void 
__launch_bounds__(M/(2*Q), 1024/(M/(2*Q)))
bmulFFT( uint32_t clgm
                       , typename P::uint_t  invM
                       , typename P::uint_t* omegas
                       , typename P::uint_t* omegas_inv
                       , typename P::uhlf_t* ass
                       , typename P::uhlf_t* bss
                       , typename P::uint_t* rss0
) {
    using uhlf_t = typename P::uhlf_t;
    
    extern __shared__ uint64_t sh_mem_char[]; // hold M*sizeof(uint_t)
    uhlf_t* shmhalf = (uhlf_t*) sh_mem_char;

    uhlf_t Ahlf[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf, ass, Ahlf);
    uhlf_t Bhlf[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf+M, bss, Bhlf);
    __syncthreads();
    
    uhlf_t Rhlf[2*Q];
    bmulFftReg<P,M,Q> ( clgm, invM, omegas, omegas_inv, Ahlf, Bhlf, shmhalf, Rhlf );
    
    uhlf_t* rss = (uhlf_t*)rss0;
    cpReg2Glb<uhlf_t,1,M,2*Q>(1, shmhalf, Rhlf, rss);
    //fromFftReg2Glb<P,(M*P::base)/(P::base+1),2*Q>( Rhreg, shmhalf, rss );
}

template<typename P, uint32_t M, uint32_t Q> __global__ void 
__launch_bounds__(M/(2*Q), 1024/(M/(2*Q)))
polyFttKer ( uint32_t clgm
                , typename P::uint_t  invM
                , typename P::uint_t* omegas
                , typename P::uint_t* omegas_inv
                , typename P::uhlf_t* ass
                , typename P::uhlf_t* bss
                , typename P::uhlf_t* rss
) {
    using uhlf_t = typename P::uhlf_t;

    extern __shared__ uint64_t sh_mem_char[]; // hold M*sizeof(uint_t)
    uhlf_t* shmhalf = (uhlf_t*) sh_mem_char;

    uhlf_t Arg[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf, ass, Arg);
    __syncthreads();

    // t1 = a*a = a^2
    uhlf_t t1[Q];
    bmulFftReg<P,M,Q> ( clgm, invM, omegas, omegas_inv, Arg, Arg, shmhalf, t1 );

    uhlf_t Brg[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf+M, bss, Brg);
    __syncthreads();
    
    // t2 = t1 + b = a^2 + b
    uhlf_t t2[2*Q];
    baddRegMul2Fft<P, M, 2*Q, 1>( shmhalf, t1, Brg, t2 );

    // t6 = a*b
    uhlf_t t6[2*Q];
    bmulFftReg<P,M,Q> ( clgm, invM, omegas, omegas_inv, Arg, Brg, shmhalf, t6 );

    // t3 = b^2
    uhlf_t t3[2*Q];
    bmulFftReg<P,M,Q> ( clgm, invM, omegas, omegas_inv, Brg, Brg, shmhalf, t3 );
    
    // t4 = t3 + b = b^2 + b
    uhlf_t t4[2*Q];
    baddRegMul2Fft<P, M, 2*Q, 1>( shmhalf, t3, Brg, t4 );

    // t5 = t2 * t4 = (a^2 + b) * (b^2 + b)
    uhlf_t t5[2*Q];
    bmulFftReg<P,M,Q> ( clgm, invM, omegas, omegas_inv, t2, t4, shmhalf, t5 );
    
    // R = t5 + t6 = (a^2 + b) * (b^2 + b) + a*b
    uhlf_t Rrg[2*Q];
    baddRegMul2Fft<P, M, 2*Q, 1>( shmhalf, t5, t6, Rrg );
    
    // write result to global memory
    cpReg2Glb<uhlf_t,1,M,2*Q>(1, shmhalf, Rrg, rss);    
}


template<typename P, uint32_t M, uint32_t Q>
__global__ void bmulFFTvalid( uint32_t clgm
                       , typename P::uint_t  invM
                       , typename P::uint_t* omegas
                       , typename P::uint_t* omegas_inv
                       , typename P::uhlf_t* ass
                       , typename P::uhlf_t* bss
                       , typename P::uint_t* rss0
) {
    using pft    = zmod_t<P>;
    using uint_t = typename P::uint_t;
    using uhlf_t = typename P::uhlf_t;
    
    extern __shared__ uint64_t sh_mem_char[];
    uint_t* shmem   = (uint_t*) sh_mem_char;
    uhlf_t* shmhalf = (uhlf_t*) sh_mem_char;

    uhlf_t Ahlf[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf, ass, Ahlf);
    //__syncthreads();
    uhlf_t Bhlf[2*Q];
    cpGlb2Reg<uhlf_t,1,M,2*Q>(1, shmhalf+M, bss, Bhlf);
    __syncthreads();
    
    uint_t Arg[2*Q], Afft[2*Q];
    for(int q=0;q<2*Q;q++) Arg[q] = Ahlf[q];
    
    fft<P,M,Q>(shmem, clgm, omegas, Arg, Afft);
    
    //////////////
    uint_t Brg[2*Q], Bfft[2*Q];
    for(int q=0;q<2*Q;q++) Brg[q] = Bhlf[q];
    fft<P,M,Q>(shmem, clgm, omegas, Brg, Bfft);
    
    uint_t Trg[2*Q], Rrg[2*Q];
    for(int i=0; i<2*Q; i++) {
        Trg[i] = pft::mul(Afft[i], Bfft[i]);
    }

    ifft<P,M,Q>(shmem, clgm, invM, omegas_inv, Trg, Rrg);

    cpReg2Glb<uint_t,1,M,2*Q>(1, shmem, Rrg, rss0);
}

#endif // KERNEL_FFT_MUL
