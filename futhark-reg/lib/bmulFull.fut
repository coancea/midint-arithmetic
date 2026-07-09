import "../intrinsics-accs"
import "types"
import "badd"
import "bmul"
import "utils"


def convolution2ndHalf [n] (Q: i64) (m:i32) (offset: i32) (k1: i32) (ash: [n]uint) (bsh: [n]uint) : [Q+2]uint =
  let accums = #[scratch] replicate Q (replicate 2 zero_uint)
  let carries= #[scratch] replicate Q zero_c
  let Q = i32.i64 Q
  let M = m * (2*Q)

  let (accums, carries) =
    loop (accums, carries) for q < Q do
      let accums[q,0] = zero_uint
      let accums[q,1] = zero_uint
      let carries[q]  = zero_c
      in  (accums, carries)
  
  let (accums, carries) =
    loop (accums, carries) for i' < M - k1 - 1 do -- for i < k1+1 do
      -- let i = i' + k1 + 1
      -- let j = M + k1 - i in
      let i = i' + k1 + 1
      let j = (M - 1) - i'
      in
      loop (accums, carries) for q < Q do
        let q_rev = (Q - 1) - q
        let (a1, a2, c) = computeIter64 offset i (j-q_rev) ash bsh (accums[q,0], accums[q,1], carries[q])
        let accums[q,0] = a1
        let accums[q,1] = a2
        let carries[q]  = c
      in  (accums, carries)
  --
  let (accums, carries) =
    loop (accums, carries) for qm1 < Q-1 do
      let q = 1 + qm1 in
      loop (accums, carries) for i < Q-q do
        let reg_ind = Q - 1 - i - q
        let (a1, a2, c) = computeIter64 offset (k1+1-q) (M-1-i) ash bsh 
                                        (accums[reg_ind,0], accums[reg_ind,1], carries[reg_ind])
        let accums[reg_ind,0] = a1
        let accums[reg_ind,1] = a2
        let carries[reg_ind]  = c
        in  (accums, carries)
  --
  in combineQ accums carries

-- Q is half the total seq factor, for example when each thread computes 4 elements, Q == 2
def wrapperConv2ndHalf (Q: i64) (n: i32) (ash: []uint) (bsh: []uint) (tid: i32) : ([Q+2]uint, [Q+2]uint) =
  -- let Q32 = i32.i64 Q
  let instance = tid / n
  let offset   = instance * ( (2 * i32.i64 Q) * n )
  let ltid     = tid % n
  --
  let q_m_ltid = ltid * i32.i64 Q
  let k1 = q_m_ltid + i32.i64 Q - 1
  let lhcs0 = convolution2ndHalf Q n offset k1 ash bsh
  --
  let k2 = ((2*i32.i64 Q)*n - 1) - q_m_ltid
  let lhcs1 = convolution2ndHalf Q n offset k2 ash bsh
  in  (lhcs0, lhcs1)

def bmulSftFullRegs [IPB][M][Q] (h:i64) (As: [IPB*M][2*Q]uint) (Bs: [IPB*M][2*Q]uint)
        : [IPB*M][2*Q]uint =
  let Areg = #[glb2reg_only(1)] manifest As
  let Breg = #[glb2reg_only(1)] manifest Bs
  let Ash = cpReg2Shm Areg -- [2*Q] registers per thread
  let Bsh = cpReg2Shm Breg -- [2*Q] registers per thread
  --
  let Hini = replicate (IPB*2) zero_uint
  let (Lsh, Hsh, Hini1) = bmulShmQ M Q Hini Ash Bsh
  let Hreg= cpShm2Reg Hsh
  let Lreg= cpShm2Reg Lsh
  --
  let (Rreg1, ipb_carries) = baddRegGen Lreg Hreg
  --
  -- add the carry to Hini1
  let fHini (Hacc: *acc ([IPB*2]uint)) (tid: i64) : acc ([IPB*2]uint) =
    let carry = uint_bool ( ipb_carries[tid,0] & 1 == 1)
    let instn = tid / M
    in  if (tid+1) % M == 0
        then write Hacc (2*instn+1) carry
        else Hacc
  let Hini1' = opaque <| reduce_by_index_stream Hini1 (+) zero_uint fHini (iota (IPB*M))
  --
  -- now do it again for the second half
  let Ash = cpReg2Shm Areg -- [2*Q] registers per thread
  let Bsh = cpReg2Shm Breg -- [2*Q] registers per thread
  let (vec_lhcs0, vec_lhcs1) = unzip <| opaque <|
       #[toregmem(1)] map (wrapperConv2ndHalf Q (i32.i64 M) Ash Bsh)
       <| map i32.i64 <| iota (IPB*M)
  let (Lsh, Hsh, _) = from4Reg2ShmQ Ash Bsh Hini1' (copy vec_lhcs0) (copy vec_lhcs1)
  let Hreg= cpShm2Reg Hsh
  let Lreg= cpShm2Reg Lsh
  --
  let Rreg2 = baddReg Lreg Hreg
  --
  -- publish the whole result to shared memory
  let shm = replicate ((IPB*M)*(2*Q)) zero_uint
  let fShift (myacc: *acc ([(IPB*M)*(2*Q)]uint)) (tid: i64) : acc ([(IPB*M)*(2*Q)]uint) =
    let instance = tid / M
    let offset   = instance * (M*(2*Q)) 
    let ltid     = tid % M in
    loop myacc for q < 2*Q do
      let ind = ltid*(2*Q) + q
      let myacc = if h <= ind
                  then write myacc (offset + ind - h) Rreg1[tid, q]
                  else myacc
      let myacc = if ind < h
                  then write myacc (offset + M*(2*Q) - h + ind) Rreg2[tid, q]
                  else myacc
      in  myacc
  let resShm = scatter_stream shm fShift (iota (IPB*M))
  in  cpShm2Reg resShm

