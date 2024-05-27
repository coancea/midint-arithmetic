#include "helper.h"
//#include "goldenSeq.h"

//#define WITH_INT_128 1

#include "ker-classic-mul.cu.h"
#include "ker-fft-mul.cu.h"

using namespace std;

#define GPU_RUNS_ADD    50
#define GPU_RUNS_MUL    25
#define ERR         0.000005

#define WITH_VALIDATION 1


template<int m, int nz>
void mkRandArrays ( int num_instances
                  , uint64_t** h_as
                  , uint64_t** h_bs
                  , uint64_t** h_rs_gmp
                  , uint64_t** h_rs_our
                  ) {

    *h_as     = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_bs     = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_rs_gmp = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_rs_our = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
        
    ourMkRandom<m, nz>(num_instances, (uint32_t*)*h_as);
    ourMkRandom<m, nz>(num_instances, (uint32_t*)*h_bs);
}

/****************************/
/*** Big-Integer Addition ***/
/****************************/

template<class Base, uint32_t m>  // m is the size of the big word in Base::uint_t units
void gpuAdd ( int num_instances
            , typename Base::uint_t* h_as
            , typename Base::uint_t* h_bs
            , typename Base::uint_t* h_rs
            ) 
{
    using uint_t = typename Base::uint_t;
    //using carry_t= typename Base::carry_t;
    
    uint_t* d_as;
    uint_t* d_bs;
    uint_t* d_rs;
    size_t mem_size_nums = num_instances * m * sizeof(uint_t);
    
    // 1. allocate device memory
    cudaMalloc((void**) &d_as, mem_size_nums);
    cudaMalloc((void**) &d_bs, mem_size_nums);
    cudaMalloc((void**) &d_rs, mem_size_nums);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, h_as, mem_size_nums, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, h_bs, mem_size_nums, cudaMemcpyHostToDevice);

    // 3. kernel dimensions
    const uint32_t q = 4; // use 8 for A4500 
    
#if 1
    const uint32_t Bprf = 256;
    const uint32_t m_lft = LIFT_LEN(m, q);
    const uint32_t ipb = ((m_lft / q) >= Bprf) ? 1 : 
                           (Bprf + (m_lft / q) - 1) / (m_lft / q);
#else
    const uint32_t m_lft = m;
    const uint32_t ipb = (128 + m/q - 1) / (m/q);
#endif
    assert(m_lft % q == 0 && m_lft >= q);
    
    dim3 block( ipb * (m_lft/q), 1, 1);
    dim3 grid ( (num_instances + ipb - 1)/ipb, 1, 1);
    
    // 4. dry run
    {
        baddKer<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    // 5. timing instrumentation
    {
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_ADD; i++) {
            baddKer<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_ADD;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed; 
        double bytes_accesses = 3.0 * num_instances * m * sizeof(uint_t);  
        double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

        printf( "Our Addition of %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigabytes, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    { // 6 additions kernel

        // 4. dry run
        {
            a6pb10Ker<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
            cudaDeviceSynchronize();
            gpuAssert( cudaPeekAtLastError() );
        }
    
        // 5. timing instrumentation
        {
            uint64_t elapsed;
            struct timeval t_start, t_end, t_diff;
            gettimeofday(&t_start, NULL); 
            
            for(int i=0; i<GPU_RUNS_ADD; i++) {
                a6pb10Ker<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
            }
            
            cudaDeviceSynchronize();

            gettimeofday(&t_end, NULL);
            timeval_subtract(&t_diff, &t_end, &t_start);
            elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_ADD;

            gpuAssert( cudaPeekAtLastError() );

            double runtime_microsecs = elapsed; 
            double bytes_accesses = 3.0 * num_instances * m * sizeof(uint_t);  
            double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

            printf( "Our SIX Additions of %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
                  , m*x*32, Base::bits, num_instances, elapsed, gigabytes, num_instances / runtime_microsecs
                  );
        }    
    }
    
    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}


template<int m>
void gmpAdd(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* rs) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpAddMulOnce<m>(true, it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

template<class Base, int m> // m is the size of the big word in u32 units
void testAddition(int num_instances
                , uint64_t* h_as_64
                , uint64_t* h_bs_64
                , uint64_t* h_rs_gmp_64
                , uint64_t* h_rs_our_64
                , uint32_t with_validation
) {
    using uint_t = typename Base::uint_t;
    
    uint_t *h_as = (uint_t*) h_as_64;
    uint_t *h_bs = (uint_t*) h_bs_64;
    uint_t *h_rs_our = (uint_t*) h_rs_our_64;
    uint32_t *h_rs_gmp_32 = (uint32_t*) h_rs_gmp_64;

    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));

    if(with_validation)
        gmpAdd<m>(num_instances, (uint32_t*)h_as, (uint32_t*)h_bs, h_rs_gmp_32);

    gpuAdd<Base, m/x>(num_instances, h_as, h_bs, h_rs_our);

#if 0
    uint32_t querry_instance = 0;
    printf("as[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_as);
    printf("bs[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_bs);
    printf("rs_gmp[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_rs_gmp);
    printf("rs_our[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_rs_our);
#endif


    if(with_validation)  
        validateExact(h_rs_gmp_32, (uint32_t*)h_rs_our, num_instances*m);
}

/***************************************************/
/*** Big-Integer Classical Multiplication O(n^2) ***/
/***************************************************/

template<class Base, int m> // m is the size of the big word in Base::uint_t
void gpuMultiply( int num_instances
                , typename Base::uint_t* h_as
                , typename Base::uint_t* h_bs
                , typename Base::uint_t* h_rs
                ) 
{
    using uint_t = typename Base::uint_t;
    uint_t* d_as;
    uint_t* d_bs;
    uint_t* d_rs;
    size_t mem_size_nums = num_instances * m * sizeof(uint_t);
    
    // 1. allocate device memory
    cudaMalloc((void**) &d_as, mem_size_nums);
    cudaMalloc((void**) &d_bs, mem_size_nums);
    cudaMalloc((void**) &d_rs, mem_size_nums);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, h_as, mem_size_nums, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, h_bs, mem_size_nums, cudaMemcpyHostToDevice);
    
    // 3. kernel dimensions; q must be 4; seq-factor = 2*q
    const uint32_t q    = 4;     // 4
    const uint32_t Bprf = 256; //256;
    //const uint32_t Bmax = 1024;
#if 1
    const uint32_t m_lft = LIFT_LEN(m, q);
    const uint32_t ipb = ((m_lft / q) >= Bprf) ? 1 : 
                           (Bprf + (m_lft / q) - 1) / (m_lft / q);    
#else
    ////const uint32_t B = 256;  // used to be this
    const uint32_t m_lft = m;
    const uint32_t B = (m <= 1024)? 256 : m/q;    
    const uint32_t ipb = (B + m/q - 1) / (m/q);
#endif

    assert( (q % 2 == 0) && (m_lft % q == 0) && (m_lft >= q ) );

    dim3 block( ipb*m_lft/q, 1, 1 );
    dim3 grid ( (num_instances+ipb-1)/ipb, 1, 1);  // BUG: it might not fit exactly!
   
#if 0 
    { // maximize the amount of shared memory for the kernel
        cudaFuncSetAttribute(bmulKer<Base,ipb,m>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        cudaFuncSetAttribute(polyKer<Base,ipb,m>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        printf( "Cosmin shmem size: %ld, B: %d, ipb: %d, num-inst: %d, m_lft: %d\n"
              , ipb*2*m_lft*sizeof(uint_t), ipb*m_lft/q, ipb, num_instances, m_lft);
    }    
#endif
    
    { // 4. dry run
        //bmulKer<Base,ipb,m><<< grid, block, ipb*2*m*sizeof(uint_t) >>>(d_as, d_bs, d_rs);
        //bmulKer<Base,ipb,m><<< grid, block >>>(d_as, d_bs, d_rs);
        bmulKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    { // 5. timing instrumentation for One Multiplication
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_MUL; i++) {
            //bmulKer<Base,ipb,m><<< grid, block >>>(d_as, d_bs, d_rs);
            bmulKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed;
        //double num_u32_ops = 4.0 * num_instances * m * m * x * x; 
        double num_u32_ops = num_instances * numAd32OpsOfMultInst<uint_t>(m);
        double gigaopsu32  = num_u32_ops / (runtime_microsecs * 1000);

        printf( "N^2 Multiplication of %d-bits Big-Numbers (in base = %d bits) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigaopsu32, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    if(1)
    { // 5. timing instrumentation for Polynomial Computation
        // dry run    
        polyKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);

        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );

    
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_MUL; i++) {
            polyKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed;
        //double num_u32_ops = 4.0 * 4.0 * num_instances * m * m * x * x; 
        double num_u32_ops = 4.0 * num_instances * numAd32OpsOfMultInst<uint_t>(m);
        double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

        printf( "Our Polynomial of %d-bits Big-Numbers (in base = %d bits) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigaopsu32, num_instances / runtime_microsecs
              );
    }

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}

template<int m>
void gmpMultiply(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* rs) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpAddMulOnce<m>(false, it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

template<typename Base, int m>  // m is the size of the big word in u32 units
void testNsqMul(  int num_instances
                , uint64_t* h_as_64
                , uint64_t* h_bs_64
                , uint64_t* h_rs_gmp_64
                , uint64_t* h_rs_our_64
                , uint32_t  with_validation
                ) {
                
    using uint_t = typename Base::uint_t;
    
    uint_t *h_as = (uint_t*)h_as_64; 
    uint_t *h_bs = (uint_t*)h_bs_64;
    uint_t *h_rs_our = (uint_t*)h_rs_our_64;
    uint32_t *h_rs_gmp_32 = (uint32_t*)h_rs_gmp_64;

    if(with_validation)
        gmpMultiply<m>(num_instances, (uint32_t*)h_as, (uint32_t*)h_bs, h_rs_gmp_32);
        
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    gpuMultiply<Base, m/x>(num_instances, h_as, h_bs, h_rs_our);
    
#if 0
    const uint32_t mm = m/x;
    uint32_t querry_instance = 0;
    printf("as[%d]: ", querry_instance);
    printInstance<uint64_t,mm>(querry_instance, (uint64_t*)h_as);
    printf("bs[%d]: ", querry_instance);
    printInstance<uint64_t,mm>(querry_instance, (uint64_t*)h_bs);
    printf("rs_gmp[%d]: ", querry_instance);
    printInstance<uint64_t,mm>(querry_instance, h_rs_gmp_64);
    printf("rs_our[%d]: ", querry_instance);
    printInstance<uint64_t,mm>(querry_instance, (uint64_t*)h_rs_our);
#endif

    if(with_validation)
        validateExact<uint32_t>(h_rs_gmp_32, (uint32_t*)h_rs_our, num_instances*m);
}

/****************************************************/
/*** Big-Integer FFT Multiplication O(n * log(n)) ***/
/****************************************************/

template<typename P, uint32_t m>
void gpuMulFFT( int num_instances
              , typename P::uhlf_t* h_as
              , typename P::uhlf_t* h_bs
              , typename P::uint_t* h_rs
              ) {
    using uint_t = typename P::uint_t;
    using uhlf_t = typename P::uhlf_t;
    
    uhlf_t* d_as;
    uhlf_t* d_bs;
    uint_t* d_rs;
    uint_t* d_omegas_inv;
    uint_t* d_omegas;
    size_t mem_one_bnum  = m * sizeof(uint_t);
    size_t mem_size_out = num_instances * mem_one_bnum;
    size_t mem_size_inp = num_instances * m * sizeof(uhlf_t);
    
    // 1. allocate device memory
    cudaMalloc((void**) &d_as, mem_size_inp);
    cudaMalloc((void**) &d_bs, mem_size_inp);
    cudaMalloc((void**) &d_rs, mem_size_out);
    cudaMalloc((void**) &d_omegas, mem_one_bnum);
    cudaMalloc((void**) &d_omegas_inv, mem_one_bnum);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, h_as, mem_size_inp, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, h_bs, mem_size_inp, cudaMemcpyHostToDevice);
    
    // 3. kernel dimensions; q must be 2!
    //const uint32_t q = 2;
    //const uint32_t B = 256;

    const uint32_t Q = ((m/2) + 1023) / 1024;

    //const uint32_t ipb = 1;
    assert(m % (Q*2) == 0 && is2pow<int>(m));
    dim3 block( m/(2*Q), 1, 1 );
    dim3 grid (num_instances, 1, 1);

/**
 * ToDo: compute: omega, omega_inv, omegas, omegas_inv and transfer the latter two to GPU space.
 */
    uint32_t clgm = ceilLg<uint32_t>(m);
    uint_t  invM = zmod_t<P>::inv(m);
    uint_t* h_omegas     = (uint_t*) malloc(m*sizeof(uint_t));
    uint_t* h_omegas_inv = (uint_t*) malloc(m*sizeof(uint_t));
    {
        uint_t omega = getOmega<P>(m);
        mkOmegas<P>(m, omega, h_omegas);
        cudaMemcpy(d_omegas, h_omegas, mem_one_bnum, cudaMemcpyHostToDevice);
        
        uint_t omega_inv = zmod_t<P>::inv(omega);
        mkOmegas<P>(m, omega_inv, h_omegas_inv);
        cudaMemcpy(d_omegas_inv, h_omegas_inv, mem_one_bnum, cudaMemcpyHostToDevice);        
    }
        
    { // maximize the amount of shared memory for the kernel
        //cudaFuncSetAttribute(bmulFFT<P, m, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        cudaFuncSetAttribute(bmulFFT<P, m, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);  // 131072 out of range
        cudaFuncSetAttribute(polyFttKer<P, m, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    }    
    
    // 4. dry run
    {
        bmulFFT<P, m, Q><<< grid, block, m*sizeof(uint_t) >>>
            ( clgm, invM, d_omegas, d_omegas_inv, d_as, d_bs, d_rs );
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
        
    // 5. timing instrumentation for One Multiplication
    {
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_MUL; i++) 
        {
            bmulFFT<P, m, Q><<< grid, block, m*sizeof(uint_t) >>>
                ( clgm, invM, d_omegas, d_omegas_inv, d_as, d_bs, d_rs );
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed;
        //double num_u32_ops = 4.0 * num_instances * m * m; 
        double num_u32_ops = num_instances * numAd32OpsOfMultInst<uint_t>(m/2);
        double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

        const uint32_t x = P::bits/32;

        printf( "Our FFT Multiplication of %lu-bits Big-Numbers (M:%u, Q:%u, base:%lu) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, shmem-size:%ld, siz_inp:%lu\n"
              , m*8*sizeof(uhlf_t), m, Q, P::base
              , num_instances, elapsed, gigaopsu32
              , m*sizeof(uint_t)
              , mem_size_inp
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );
    
    // polyomial computation:
    {   
        // 4. dry run
        {
            polyFttKer<P, m, Q><<< grid, block, m*sizeof(uint_t) >>>
                ( clgm, invM, d_omegas, d_omegas_inv, d_as, d_bs, (uhlf_t*)d_rs );
            cudaDeviceSynchronize();
            gpuAssert( cudaPeekAtLastError() );
        }
            
        // 5. timing instrumentation for One Multiplication
        {
            uint64_t elapsed;
            struct timeval t_start, t_end, t_diff;
            gettimeofday(&t_start, NULL); 
            
            for(int i=0; i<GPU_RUNS_MUL; i++) 
            {
                polyFttKer<P, m, Q><<< grid, block, m*sizeof(uint_t) >>>
                    ( clgm, invM, d_omegas, d_omegas_inv, d_as, d_bs, (uhlf_t*)d_rs );
            }
            
            cudaDeviceSynchronize();

            gettimeofday(&t_end, NULL);
            timeval_subtract(&t_diff, &t_end, &t_start);
            elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

            gpuAssert( cudaPeekAtLastError() );

            double runtime_microsecs = elapsed;
            double num_u32_ops = 4.0 * num_instances * numAd32OpsOfMultInst<uint_t>(m/2);
            double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

            const uint32_t x = P::bits/32;

            printf( "Our Poly-FFT Computation of %lu-bits Big-Numbers (M:%u, Q:%u, base:%lu) runs \
%d instances in: %lu microsecs, Gu32ops/sec: %.2f, shmem-size:%ld, siz_inp:%lu\n"
                  , m*8*sizeof(uhlf_t), m, Q, P::base
                  , num_instances, elapsed, gigaopsu32
                  , m*sizeof(uint_t)
                  , mem_size_inp
                  );
        }    
    }

    free(h_omegas);
    free(h_omegas_inv);
    cudaFree(d_omegas);
    cudaFree(d_omegas_inv);
    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}

template<typename Base, int m>
void testFftMul ( int num_instances
                , uint64_t* h_as_64
                , uint64_t* h_bs_64
                , uint64_t* h_rs_gmp_64
                , uint64_t* h_rs_our_64
                , uint32_t  with_validation
                )
{
    using uint_t = typename Base::uint_t;
    using uhlf_t = typename Base::uhlf_t;
    
    uhlf_t *h_as     = (uhlf_t*) h_as_64;
    uhlf_t *h_bs     = (uhlf_t*) h_bs_64;
    uint_t *h_rs_our = (uint_t*) h_rs_our_64;
    uint32_t *h_rs_gmp = (uint32_t*) h_rs_gmp_64;

    if(with_validation)
        gmpMultiply<m>(num_instances, (uint32_t*)h_as, (uint32_t*)h_bs, h_rs_gmp);
        
    //const uint32_t x = Base::bits/32;
    const uint32_t x = 4 / sizeof(uhlf_t); //Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0) );
    
    gpuMulFFT<Base, m*x>(num_instances, h_as, h_bs, h_rs_our);
    
    if(with_validation)
        validateExact<uint32_t>(h_rs_gmp, (uint32_t*)h_rs_our, num_instances*m);
}

template<typename Base, int m>
void partValidFftMul ( typename Base::uhlf_t* h_as
                     , typename Base::uhlf_t* h_bs
                     , typename Base::uint_t* h_rs_ref
                     )
{
    using uint_t = typename Base::uint_t;
    using uhlf_t = typename Base::uhlf_t;
    
    uint_t* h_rs_our = (uint_t*) malloc( m * sizeof(uint_t) ); 
    
    gpuMulFFT<Base, m>(1, h_as, h_bs, h_rs_our);

    //printInstance<uhlf_t,m>(0, (uhlf_t*)h_rs_our);

    validateExact<uint_t>(h_rs_ref, h_rs_our, m);
    
    free(h_rs_our);
}

void runPartValidationFFT() {
    //uint32_t inp_inv[] = { 2170333833, 97506133, 2151463229, 2106724456, 1111720637, 2039631599, 1651518977, 2515781039, 53415978, 3097490970, 451002497, 1928198602, 2032731680, 2298827368, 1582725215, 2780389298};
    
    using uhlf_t = typename FftPrime32::uhlf_t;
    using uint_t = typename FftPrime32::uint_t;
    
    uhlf_t inp_a[] =   { 11400, 28374, 23152, 9576
                       , 29511, 20787, 13067, 14015
                       , 0, 0, 0, 0
                       , 0, 0, 0, 0 
                       };
    uhlf_t inp_b[] =   { 30268, 20788, 8033, 15446
                       , 26275, 11619,  2494,  7016
                       , 0, 0, 0, 0
                       , 0, 0, 0, 0
                       };

    uint_t ref[]   =   { 345055200, 1095807432, 1382179648, 1175142886
                       , 2016084656, 2555168834, 2179032777, 1990011337
                       , 1860865174, 1389799087, 942120918, 778961552
                       , 341270975, 126631482, 98329240, 0
                       };
    
    partValidFftMul<FftPrime32, 16>(inp_a, inp_b, ref);
}

/////////////////////////////////////////////////////////
// Main program that runs test suits
/////////////////////////////////////////////////////////
 
template<typename Base>
void runAdditions(uint64_t total_work) {
    uint64_t *h_as, *h_bs, *h_rs_gmp, *h_rs_our;
    mkRandArrays<32,32>( total_work/32, &h_as, &h_bs, &h_rs_gmp, &h_rs_our );
    
#if 1
    testAddition<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base, 1024>( total_work/1024, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,  512>( total_work/512,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,  256>( total_work/256,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,  128>( total_work/128,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,   64>( total_work/64,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,   32>( total_work/32,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testAddition<Base,   16>( total_work/16,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
#endif
    free(h_as);
    free(h_bs);
    free(h_rs_gmp);
    free(h_rs_our);
}
 
template<typename Base>
void runNaiveMuls(uint64_t total_work) {
    using uint_t = typename Base::uint_t;
    uint64_t *h_as, *h_bs, *h_rs_gmp, *h_rs_our;
    mkRandArrays<32,32>( total_work/32, &h_as, &h_bs, &h_rs_gmp, &h_rs_our );

//    testNsqMul<Base, 512*3>( total_work/(512*3), h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );

//    testNsqMul<Base, 2048>( total_work/(2048), h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION ); 
#if 1
    testNsqMul<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );

    testNsqMul<Base, 1024>( total_work/1024, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    
    testNsqMul<Base,  512>( total_work/512,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,  256>( total_work/256,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,  128>( total_work/128,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   64>( total_work/64,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   32>( total_work/32,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   16>( total_work/16,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
#endif
    free(h_as);
    free(h_bs);
    free(h_rs_gmp);
    free(h_rs_our);    
}

template<typename Base>
void runFFTMuls(uint64_t total_work) {
    using uint_t = typename Base::uint_t;
    uint64_t *h_as, *h_bs, *h_rs_gmp, *h_rs_our;
    
    //total_work = 4096*512;
    mkRandArrays<32,32>( total_work/16, &h_as, &h_bs, &h_rs_gmp, &h_rs_our );

    //testFftMul<Base, 4096*2>( total_work/(4096*2), h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    
    //testFftMul<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    
    //testFftMul<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    
    //testFftMul<Base, 4096*4>( 8, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );

#if 1
    //testFftMul<Base, 4096*4>( total_work/(4096*4), h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testFftMul<Base, 4096*2>( total_work/(4096*2), h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base, 1024>( total_work/1024, h_as, h_bs, h_rs_gmp, h_rs_our, 0 );

    testFftMul<Base,  512>( total_work/512,  h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base,  256>( total_work/256,  h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base,  128>( total_work/128,  h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base,   64>( total_work/64,   h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base,   32>( total_work/32,   h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
    testFftMul<Base,   16>( total_work/16,   h_as, h_bs, h_rs_gmp, h_rs_our, 0 );
#endif
    
    free(h_as);
    free(h_bs);
    free(h_rs_gmp);
    free(h_rs_our);    
}


int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <batch-size>\n", argv[0]);
        exit(1);
    }
        
    const int total_work = atoi(argv[1]);

    cudaSetDevice(1);
    //testBasicOps<uint32_t> ( 1000000, 100 );
    //testBasicOps<uint64_t> ( 1000000, 100 );

    //runPartValidationFFT();

    runAdditions<U64bits>(total_work);
    runNaiveMuls<U64bits> (total_work);
    //runFFTMuls<FftPrime64>(total_work);
    runFFTMuls<FftPrime32>(total_work);

#if 0       
    runAdditions<U32bits>(total_work);
    runAdditions<U64bits>(total_work);
    
    runNaiveMuls<U32bits>(total_work);
    runNaiveMuls<U64bits>(total_work);
#endif

}
