import "../intrinsics-accs"
import "types"
import "badd"

---------------------------------------------------------------
--- Utilities
---------------------------------------------------------------

def cpShm2Reg [M][Q] 't (Ash: [M*Q]t) : *[M][Q]t =
  #[unsafe]
  let ff tid = #[sequential] map (\q -> Ash[tid*Q + q]) (iota Q)
  in  opaque <| #[toregmem(1)] map ff (iota M)

def cpShm2RegPad [M][Q] 't (m: i64) (pad: t) (Ash: [M*Q]t) : *[M][Q]t =
  #[unsafe]
  let f tid j = let ind = tid*Q + j in if ind < m then Ash[ind] else pad
  let ff tid = #[sequential] map (f tid) (iota Q)
  in  opaque <| #[toregmem(1)] map ff (iota M)

def cpReg2Shm [M][Q] 't (Areg: [M][Q]t) : *[M*Q]t =
  #[unsafe]
  let Ash = #[scratch] replicate (M*Q) Areg[0,0]
  let f (Aacc: *acc ([M*Q]t)) (tid: i64) : acc ([M*Q]t) =
    loop Aacc for q < Q do
      write Aacc (tid*Q + q) (Areg[tid][q])
  in scatter_stream Ash f (iota M)

def cpReg2ShmNoAcc [M][Q] (Areg: [M][Q]uint) : *[M*Q]uint =
  #[unsafe]
  let f (row : [Q]uint) =
    loop res = #[scratch] replicate Q (row[0]) for q < Q do
      let res[q] = row[q] in res
  in flatten (map f Areg)

def from4Reg2ShmQ_OLD [IPB][N][Q]
        (Lsh:   *[(IPB*N)*(2*Q)]uint)
        (Hsh:   *[(IPB*N)*(2*Q)]uint)
        (iniH:  *[IPB*2]uint)
        (lhcs0: *[IPB*N][Q+2]uint) 
        (lhcs1: *[IPB*N][Q+2]uint)
      : ([IPB*N][2*Q]uint, [IPB*N][2*Q]uint, [IPB*2]uint) =
  #[unsafe]
--  let Lsh = opaque <| replicate ((IPB*N)*(2*Q)) zero_uint
--  let Hsh = opaque <| replicate ((IPB*N)*(2*Q)) zero_uint
  --
  let fH (Hacc: *acc ([(IPB*N)*(2*Q)]uint)) (tid: i64) : acc ([(IPB*N)*(2*Q)]uint) =
    #[unsafe]
    let instance = tid / N
    let ltid     = tid % N
    let offset   = instance * ( N * (Q * 2) )
    --
    let twoltid = offset + Q*ltid
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zero_uint    
    let Hacc = write Hacc (twoltid+Q)   (lhcs0[tid,Q])
    let Hacc = write Hacc (twoltid+Q+1) (lhcs0[tid,Q+1])
    --
    let n_m_2ltid = offset + (2*Q)*N - Q*ltid - Q
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zero_uint
    let (high, carry, ind) =
        if ltid == 0
        then (iniH[instance*2], iniH[instance*2+1], offset) -- (0, 0, offset)
        else (lhcs1[tid,Q], lhcs1[tid,Q+1], n_m_2ltid + Q) 
    let Hacc = write Hacc ind high
    let Hacc = write Hacc (ind+1) carry
    in  Hacc
  let Hsh = scatter_stream Hsh fH (iota (IPB*N))
  --
  let fLastH (LHacc: *acc ([IPB*2]uint)) (tid: i64) : acc ([IPB*2]uint) =
    let instance = tid / N
    let ltid     = tid % N in
    if ltid == 0
    then let LHacc = write LHacc (2*instance+0) lhcs1[tid][Q]
         let LHacc = write LHacc (2*instance+1) lhcs1[tid][Q+1]
         in  LHacc
    else LHacc
  let iniH' = scatter_stream iniH fLastH (iota (IPB*N))
  --
  let fL (Lacc: *acc ([(IPB*N)*(2*Q)]uint)) (tid: i64) : acc ([(IPB*N)*(2*Q)]uint) =
    #[unsafe]
    let instance = tid / N
    let ltid     = tid % N
    let offset   = instance * ( N * (Q * 2) )
    --
    let twoltid = offset + Q*ltid
    let Lacc = loop Lacc for q < Q do write Lacc (twoltid+q) (lhcs0[tid,q])
    let n_m_2ltid = offset + (2*Q)*N - Q*ltid - Q
    in loop Lacc for q < Q do write Lacc (n_m_2ltid + q) (lhcs1[tid,q])
  let Lsh = opaque <| scatter_stream Lsh fL (iota (IPB*N))
  --
  let Hreg= cpShm2Reg Hsh
  let Lreg= cpShm2Reg Lsh
  in  (Lreg, Hreg, iniH')

---------------------
---------------------

def from4Reg2ShmQ [x][IPB][N][Q]
        (Lsh:   *[x]uint)
        (Hsh:   *[x]uint)
        (iniH:  *[IPB*2]uint)
        (lhcs0: *[IPB*N][Q+2]uint) 
        (lhcs1: *[IPB*N][Q+2]uint)
      : ([x]uint, [x]uint, [IPB*2]uint) =
  #[unsafe]
  let fH (Hacc: *acc ([x]uint)) (tid: i64) : acc ([x]uint) =
    #[unsafe]
    let instance = tid / N
    let ltid     = tid % N
    let offset   = instance * ( N * (Q * 2) )
    --
    let twoltid = offset + Q*ltid
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zero_uint    
    let Hacc = write Hacc (twoltid+Q)   (lhcs0[tid,Q])
    let Hacc = write Hacc (twoltid+Q+1) (lhcs0[tid,Q+1])
    --
    let n_m_2ltid = offset + (2*Q)*N - Q*ltid - Q
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zero_uint
    let (high, carry, ind) =
        if ltid == 0
        then (iniH[instance*2], iniH[instance*2+1], offset) -- (0, 0, offset)
        else (lhcs1[tid,Q], lhcs1[tid,Q+1], n_m_2ltid + Q) 
    let Hacc = write Hacc ind high
    let Hacc = write Hacc (ind+1) carry
    in  Hacc
  let Hsh = scatter_stream Hsh fH (iota (IPB*N))
  --
  let fLastH (LHacc: *acc ([IPB*2]uint)) (tid: i64) : acc ([IPB*2]uint) =
    let instance = tid / N
    let ltid     = tid % N in
    if ltid == 0
    then let LHacc = write LHacc (2*instance+0) lhcs1[tid][Q]
         let LHacc = write LHacc (2*instance+1) lhcs1[tid][Q+1]
         in  LHacc
    else LHacc
  let iniH' = scatter_stream iniH fLastH (iota (IPB*N))
  --
  let fL (Lacc: *acc ([x]uint)) (tid: i64) : acc ([x]uint) =
    #[unsafe]
    let instance = tid / N
    let ltid     = tid % N
    let offset   = instance * ( N * (Q * 2) )
    --
    let twoltid = offset + Q*ltid
    let Lacc = loop Lacc for q < Q do write Lacc (twoltid+q) (lhcs0[tid,q])
    let n_m_2ltid = offset + (2*Q)*N - Q*ltid - Q
    in loop Lacc for q < Q do write Lacc (n_m_2ltid + q) (lhcs1[tid,q])
  let Lsh = opaque <| scatter_stream Lsh fL (iota (IPB*N))
  --
  in (Lsh, Hsh, iniH') 


def combine2Scals (l0:uint, h1:uint, c2:uint_c) (l1:uint, h2:uint, c3:uint_c) : (uint,uint,uint,uint) =
  let l1' = l1 + h1
  let c2' = c2 + c_bool (l1' < l1) -- we assume carry is big enough to not overflow
  let h2' = h2 + uint_c c2'
  let c3' = c3 + c_bool (h2' < h2)
  in  (l0, l1', h2', uint_c c3')


let computeIter64 (offset: i32) (i: i32) (j: i32) 
                  (ash: []uint) (bsh: []uint) 
                  (l: uint, h: uint, c: uint_c) : (uint, uint, uint_c) =
  let ai = #[unsafe] ash[offset+i]
  let bj = #[unsafe] bsh[offset+j]
  let ck_l = ai * bj
  let n_l = l + ck_l
  let c_l = uint_bool ( (c_uint (n_l >> size_c)) < (c_uint (ck_l >> size_c)) )
  let n_h = h + c_l
  let ck_h = uint_mul_hi ai bj
  let n_h = n_h + ck_h
  let c_h = c_bool ( (c_uint (n_h >> size_c)) < (c_uint (h >> size_c)) )
  let n_c = c + c_h
  in  (n_l, n_h, n_c)

def combineQ [Q] (accums: [Q][2]uint) (carries: [Q]uint_c) : [Q+2]uint =
  #[unsafe]
  let lhcs = #[scratch] replicate (Q+2) zero_uint
  let lhcs[0] = accums[0,0]
  let h_res = accums[0,1]
  let c_res = carries[0]
  let (lhcs, h_res, c_res) = #[unsafe]
    loop (lhcs, h_res, c_res)
      for qm1 < Q-1 do
        let q = qm1 + 1
        let l = accums[q,0]
        let h = accums[q,1]
        let lhcs[q] = l + h_res
        let h_res = h + uint_c (c_res + c_bool (lhcs[q] < l))
        let c_res = carries[q] + c_bool (h_res < h)
        in  (lhcs, h_res, c_res)
  let lhcs[Q]   = h_res
  let lhcs[Q+1] = uint_c c_res
  in  lhcs

def convolutionQ [n] (Q: i64) (offset: i32) (k1: i32) (ash: [n]uint) (bsh: [n]uint) : [Q+2]uint =
  #[unsafe]
  let accums = #[scratch] replicate Q (replicate 2 zero_uint)
  let carries= #[scratch] replicate Q zero_c
  let Q = i32.i64 Q

  let (accums, carries) =
    loop (accums, carries) for q < Q do
      let accums[q,0] = zero_uint
      let accums[q,1] = zero_uint
      let carries[q]  = zero_c
      in  (accums, carries)
  
  let (accums, carries) =
    loop (accums, carries) for i < k1+1 do
      let j = k1 - i in
      loop (accums, carries) for q < Q do
        let (a1, a2, c) = computeIter64 offset i (j+q) ash bsh (accums[q,0], accums[q][1], carries[q])
        let accums[q,0] = a1
        let accums[q,1] = a2
        let carries[q]  = c
      in  (accums, carries)
  --
  let (accums, carries) =
    loop (accums, carries) for qm1 < Q-1 do
      let q = qm1 + 1 in
      loop (accums, carries) for i < Q-q do
        let (a1, a2, c) = computeIter64 offset (k1+q) i ash bsh (accums[i+q,0], accums[i+q,1], carries[i+q])
        let accums[i+q,0] = a1
        let accums[i+q,1] = a2
        let carries[i+q]  = c
        in  (accums, carries)
  --
  in combineQ accums carries


-- Q is half the total seq factor, for example when each thread computes 4 elements, Q == 2
def wrapperConvQ (Q: i64) (n: i32) (ash: []uint) (bsh: []uint) (tid: i32) : ([Q+2]uint, [Q+2]uint) =
  -- let Q32 = i32.i64 Q
  let instance = tid / n
  let offset   = instance * ( (2 * i32.i64 Q) * n )
  let ltid     = tid % n
  --
  let k1 = i32.i64 Q * ltid
  let lhcs0 = convolutionQ Q offset k1 ash bsh
  --
  let k2 = ((2*i32.i64 Q)*n - i32.i64 Q) - k1
  let lhcs1 = convolutionQ Q offset k2 ash bsh
  in  (lhcs0, lhcs1)

def bmulShmQ [n] [IPB]
      (M: i64) (Q: i64) (Hini: *[IPB*2]uint)
      (Ash: *[n]uint) (Bsh: *[n]uint)
    : ([n]uint, [n]uint, [IPB*2]uint) =
  let (vec_lhcs0, vec_lhcs1) =
    unzip <| opaque <|
    #[toregmem(1)] map (wrapperConvQ Q (i32.i64 M) Ash Bsh)
    <| map i32.i64 <| iota (IPB*M)
  --
  in  from4Reg2ShmQ Ash Bsh Hini (copy vec_lhcs0) (copy vec_lhcs1)

def bmulRegsQ [IPB][M][Q] (Areg: [IPB*M][2*Q]uint) (Breg: [IPB*M][2*Q]uint) : [IPB*M][2*Q]uint = 
  #[unsafe]
  let Ash = cpReg2Shm Areg -- this works in double Q
  let Bsh = cpReg2Shm Breg -- this works in double Q
  --
  let Hini = replicate (IPB*2) zero_uint
  let (Lsh, Hsh, _) = bmulShmQ M Q Hini Ash Bsh
  --
  let Hreg= cpShm2Reg Hsh
  let Lreg= cpShm2Reg Lsh
----
--  let (vec_lhcs0, vec_lhcs1) =
--    unzip <| opaque <|
--    #[toregmem(1)] map (wrapperConvQ Q (i32.i64 M) Ash Bsh)
--    <| map i32.i64 <| iota (IPB*M)
----
--  let (Lreg, Hreg, _) = 
--    -- from4Reg2ShmQ vec_lhcs0 vec_lhcs1
--    from4Reg2ShmQ Ash Bsh (replicate (IPB*2) zero_uint) (copy vec_lhcs0) (copy vec_lhcs1)
----
  let Rreg = baddReg Lreg Hreg
  in  Rreg

def bmul [ipb][n][q] (As: [ipb*n][2*q]uint) (Bs: [ipb*n][2*q]uint) : [ipb*n][2*q]uint =
  #[unsafe]
  let Areg = #[glb2reg_only(1)] manifest As
  let Breg = #[glb2reg_only(1)] manifest Bs
  let Rreg = bmulRegsQ Areg Breg
  in  opaque Rreg

