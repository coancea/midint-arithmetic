#ifndef KERNEL_ADDITION
#define KERNEL_ADDITION

#include "ker-helpers.cu.h"

/****************************/
/*** Prefix-Sum Operators ***/
/****************************/

/**
 * Generic Carry operator for addition that can be
 *    instantiated over numeric-basic types, such as
 *    uint8_t, uint16_t, uint32_t, uint64_t.
 *
 * The "readable" operator having neutral element (0,1)
 * can be defined in Futhark as:
 *
 *    let badd_op (ov1 : bool, mx1: bool) (ov2 : bool, mx2: bool) =
 *        ( (ov1 && mx2) || ov2,   mx1 && mx2 )
 * 
 * We shrink the implementation to encode the tuple
 *   in one integer, which has the format:
 *       last digit set      => overfolow
 *       ante-last digit set => one unit away from overflowing   
 * Hence its neutral element is 2 (one unit away from overflow):
 *
 *     let badd_op (c1: CT) (c2: CT) : CT =
 *       (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1)
 */
template<class CT>
class CarryBop {
  public:
    typedef CT InpElTp;
    typedef CT RedElTp;
    static const bool commutative = false;
    static __device__ __host__ inline CT identInp()           { return (CT)0; }
    static __device__ __host__ inline CT mapFun(const CT& el) { return el;    }
    static __device__ __host__ inline CT identity()           { return (CT)2; }
    
    static __device__ __host__ inline 
    CT apply(const CT c1, const CT c2) { 
        return ( (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1) );
    }

    static __device__ __host__ inline bool equals(const CT t1, const CT t2) { return (t1 == t2); }
    static __device__ __host__ inline CT remVolatile(volatile CT& t)   { CT res = t; return res; }
};


template<class CT>
class CarrySegBop {
  public:
    typedef CT InpElTp;
    typedef CT RedElTp;
    static const bool commutative = false;
    static __device__ __host__ inline CT identInp()           { return (CT)0; }
    static __device__ __host__ inline CT mapFun(const CT& el) { return el;    }
    static __device__ __host__ inline CT identity()           { return (CT)2; }

#if 0
    static __device__ __host__ inline 
    CT apply(const CT c1, const CT c2) { 
        return ( (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1) );
    }
#else    
    static __device__ __host__ inline 
    CT apply(const CT c1, const CT c2) {
        CT res;
        if (c2 & 4) {
            res = c2;
        } else {
            res = ( (c1 & (c2 >> 1)) | c2 ) & 1;
            res = res | (c1 & c2  & 2);
            res = res | ( (c1 | c2) & 4 );
        }
        return res;
    }
#endif
    static __device__ __host__ inline bool equals(const CT t1, const CT t2) { return (t1 == t2); }
    static __device__ __host__ inline CT remVolatile(volatile CT& t)   { CT res = t; return res; }
};


/***************************************/
/*** Warp and Block-level Prefix Sum ***/
/***************************************/

template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) ptr[idx] = OP::apply(ptr[idx-p], ptr[idx]);
        __syncwarp();
    }
    return OP::remVolatile(ptr[idx]);
}

template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    
    if(blockDim.x <= 32) { // block < WARP optimization
        return res;
    }
    
    __syncthreads();

    // 2. place the end-of-warp results in the first warp. 
    if (lane == (WARP-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }
    __syncthreads();
    // 5. publish to shared memory
    ptr[idx] = res;
    __syncthreads();
    
    return res;
}


/*********************************************/
/*** Main function for big-number addition ***/
/*********************************************/

template<class D, class S, class CT, uint32_t m, uint32_t q, D HIGHEST>
__device__ void baddRegs( volatile CT* Csh
                         , D Arg[q]
                         , S Brg[q]
                         , D rs[q]
                         ) {
    //D  rs[q];
    CT cs[q];
    
    // 1. map: add the digits pairwise, build the 
    //         partial results and the carries, and
    //         print carries to shmem
    {
        CT accum = CarrySegBop<CT>::identity();
        for(int i=0; i<q; i++) {
            uint32_t ind = threadIdx.x * q + i;
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            rs[i] = a + (D)b;
            c = (CT) ( (rs[i] < a) );
            c = c | ((rs[i] == HIGHEST) << 1);
            //c = c | ( ((ind % m) == 0) << 2 );
            if( (ind % m) == 0 )
                c = c | 4;
            
            accum = CarrySegBop<CT>::apply(accum, c);
            cs[i] = c;
        }
        Csh[threadIdx.x] = accum;
    }
    
    __syncthreads();
   
    // 2. scan the carries
    scanIncBlock< CarrySegBop<CT> >(Csh, threadIdx.x);
        
    // 3. compute the final result by adding the carry from the previous element
    {
        CT carry = CarrySegBop<CT>::identity();
        if(threadIdx.x > 0) {
            carry = Csh[threadIdx.x - 1];
        }
        //CT carry = prefix;
        for(int i=0; i<q; i++) {
            // uint32_t c = ( (carry & 1) == 1 );
            if( (cs[i] & 4) == 0 )
                rs[i] += (carry & 1);
            carry = CarrySegBop<CT>::apply(carry, cs[i]);         
        }
    }
    __syncthreads();
}

template<class uint_t, class S, class CT, uint32_t m, uint32_t q, uint_t HIGHEST>
__device__ void baddRegs1( volatile CT* sh_mem
                         , uint_t ass[q]
                         , S bss[q]
                         , uint_t rss[q]
                         ) {
    CT css[q];
    uint_t acc = CarrySegBop<CT>::identity();
    for(int i=0; i<q; i++) {
        rss[i] = ass[i] + ((uint_t)bss[i]);
        css[i] = ((uint_t) (rss[i] < ass[i])) | (((uint_t) (rss[i] == HIGHEST)) << 1);
        acc = CarrySegBop<CT>::apply(acc, css[i]);
    }
    
    uint_t last_carry = (threadIdx.x % (m/q) == 0) ? (acc | 4) : acc;
    sh_mem[threadIdx.x] = last_carry;
    __syncthreads();
    scanIncBlock< CarrySegBop<CT> >(sh_mem, threadIdx.x);
    uint_t carry_prefix = (threadIdx.x % (m/q) == 0) ? CarrySegBop<CT>::identity() : sh_mem[threadIdx.x-1];
    __syncthreads();
    
    for(int i=0; i<q; i++) {
        rss[i] += (uint_t)(carry_prefix & 1);
        carry_prefix = CarrySegBop<CT>::apply(carry_prefix, css[i]);
    }
}

/**************************************/
/*** One Big-Number Addition Kernel ***/
/**************************************/

template<typename Base, uint32_t IPB, uint32_t M, uint32_t Q>
__global__ void baddKer ( uint32_t num_instances
                        , typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        )
{
    using uint_t = typename Base::uint_t;
    using carry_t= typename Base::carry_t;
    
    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;
    
    __shared__ uint_t shmem[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)shmem;
    
    uint_t Arg[Q];
    uint_t Brg[Q];
    uint_t Rrg[Q];
    
    const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
    
    // 1. read from global to registers
    cpGlb2Reg<uint_t,IPB,M,Q>(ipb, shmem, ass, Arg);
    __syncthreads();
    cpGlb2Reg<uint_t,IPB,M,Q>(ipb, shmem, bss, Brg); 
    __syncthreads();
          
    // 2. perform the addition in fast memory
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, Arg, Brg, Rrg );
    
    // 3. write from shared to global memory
    cpReg2Glb<uint_t,IPB,M,Q>(ipb, shmem, Rrg, rss);
}

/**************************************/
/*** Six Big-Number Addition Kernel ***/
/**************************************/

template<typename Base, uint32_t IPB, uint32_t M, uint32_t Q>
__global__ void a6pb10Ker( uint32_t num_instances
                         , typename Base::uint_t* ass
                         , typename Base::uint_t* bss
                         , typename Base::uint_t* rss
                         ) 
{
    using uint_t = typename Base::uint_t;
    using carry_t= typename Base::carry_t;

    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;

    __shared__ uint_t shmem[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)shmem;
    
    const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
    
    uint_t Arg[Q];
    uint_t Brg[Q];
      
    // 1. read from global to register memory
    cpGlb2Reg<uint_t,IPB,M,Q>(ipb, shmem, ass, Arg);
    __syncthreads();
    cpGlb2Reg<uint_t,IPB,M,Q>(ipb, shmem, bss, Brg); 
    __syncthreads(); 

#if 0
    for(int i=0; i<10; i++) {
        baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, Arg, Brg, Brg);
        __syncthreads();
    }
    cpReg2Glb<uint_t,IPB,M,Q>(ipb, shmem, Brg, rss);
#endif

    // 2. perform six additions as follows:
    // t1 = a + b
    uint_t t1rg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, Arg, Brg, t1rg);

    // t2 = t1 + t1 = 2a + 2b
    uint_t t2rg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, t1rg, t1rg, t2rg);

    // t3 = t2 + b = 2a + 3b
    uint_t t3rg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, t2rg, Brg, t3rg);

    // t4 = t3 + t3 = 4a + 6b
    uint_t t4rg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, t3rg, t3rg, t4rg);

    // t5 = t4 + t3 = (4a + 6b) + (2a + 3b) = 6a + 9b
    uint_t t5rg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, t4rg, t3rg, t5rg);

    // res = t5 + b = 6a + 10b
    uint_t resrg[Q];
    baddRegs<uint_t,uint_t,carry_t,M_lft,Q,Base::HIGHEST>( carry_shm, t5rg, Brg, resrg);

    // 3. write from shared to global memory
    cpReg2Glb<uint_t,IPB,M,Q>(ipb, shmem, resrg, rss);
}


#endif //KERNEL_ADDITION
