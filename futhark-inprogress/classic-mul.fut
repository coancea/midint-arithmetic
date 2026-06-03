import "intrinsics-accs"
import "badd"

let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap3 as bs cs f = map3 f as bs cs
let imap2Intra as bs f = #[incremental_flattening(only_intra)] map2 f as bs

type D = u64
let D_mul_hi = u64.mul_hi
let D_bool = u64.bool
let zeroD = 0u64
type S = u32
let zeroS = 0u32
let lenS = 32u64
let S_bool = u32.bool
let S_D = u32.u64
let D_S = u64.u32

type Dx4   = (D,D,D,D)
type i64x4 = (i64,i64,i64,i64)

---------------------------------------------------------------
--- Utilities
---------------------------------------------------------------

def cpShm2Reg [M][Q] 't (Ash: [M*Q]t) : *[M][Q]t = #[unsafe]
  let ff tid = #[sequential] map (\q -> Ash[tid*Q + q]) (iota Q)
  in  opaque <| #[toregmem(1)] map ff (iota M)

def cpReg2Shm [M][Q] 't (Areg: [M][Q]t) : *[M*Q]t = #[unsafe]
  let Ash = #[scratch] replicate (M*Q) Areg[0,0]
  let f (Aacc: *acc ([M*Q]t)) (tid: i64) : acc ([M*Q]t) =
    loop Aacc for q < Q do
      write Aacc (tid*Q + q) (Areg[tid][q])
  in scatter_stream Ash f (iota M)

def cpReg2ShmNoAcc [M][Q] (Areg: [M][Q]D) : *[M*Q]D = #[unsafe]
  let f (row : [Q]D) =
    loop res = #[scratch] replicate Q (row[0]) for q < Q do
      let res[q] = row[q] in res
  in flatten (map f Areg)

def from4Reg2ShmQ [IPB][N][Q]
        (Lsh:  *[(IPB*N)*(2*Q)]D)
        (Hsh:  *[(IPB*N)*(2*Q)]D)
        (lhcs0: [IPB*N][Q+2]D) 
        (lhcs1: [IPB*N][Q+2]D)
      : ([IPB*N][2*Q]D, [IPB*N][2*Q]D) =
  #[unsafe]
--  let Lsh = opaque <| replicate ((IPB*N)*(2*Q)) zeroD
--  let Hsh = opaque <| replicate ((IPB*N)*(2*Q)) zeroD
  --
  let fH (Hacc: *acc ([(IPB*N)*(2*Q)]D)) (tid: i64) : acc ([(IPB*N)*(2*Q)]D) =
    #[unsafe]
    let instance = tid / N
    let ltid     = tid % N
    let offset   = instance * ( N * (Q * 2) )
    --
    let twoltid = offset + Q*ltid
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zeroD    
    let Hacc = write Hacc (twoltid+Q)   (lhcs0[tid,Q])
    let Hacc = write Hacc (twoltid+Q+1) (lhcs0[tid,Q+1])
    --
    let n_m_2ltid = offset + (2*Q)*N - Q*ltid - Q
    let Hacc = loop Hacc for qm2 < Q-2 do write Hacc (twoltid+qm2+2) zeroD
    let (high, carry, ind) =
        if ltid == 0
        then (0, 0, offset)
        else (lhcs1[tid,Q], lhcs1[tid,Q+1], n_m_2ltid + Q) 
    let Hacc = write Hacc ind high
    let Hacc = write Hacc (ind+1) carry
    in  Hacc
  let Hsh = opaque <| scatter_stream Hsh fH (iota (IPB*N))
  --
  let fL (Lacc: *acc ([(IPB*N)*(2*Q)]D)) (tid: i64) : acc ([(IPB*N)*(2*Q)]D) =
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

  let Hreg= cpShm2Reg <| opaque <| Hsh
  let Lreg= cpShm2Reg <| opaque <| Lsh
  --
  in  (opaque Lreg, opaque Hreg)

--
--def combine2 (l0:D, h1:D, c2:S) (l1:D, h2:D, c3:S) : Dx4 =
--  let l1' = l1 + h1
--  let c2' = c2 + S_bool (l1' < l1) -- we assume carry is big enough to not overflow
--  let h2' = h2 + D_S c2'
--  let c3' = c3 + S_bool (h2' < h2)
--  in  (l0, l1', h2', D_S c3')
--

let computeIter64 (offset: i32) (i: i32) (j: i32) (ash: []D) (bsh: []D) (l: D, h: D, c: S) : (D, D, S) =
  let ai = #[unsafe] ash[offset+i]
  let bj = #[unsafe] bsh[offset+j]
  let ck_l = ai * bj
  let n_l = l + ck_l
  let c_l = D_bool ( (S_D (n_l >> lenS)) < (S_D (ck_l >> lenS)) )
  let n_h = h + c_l
  let ck_h = D_mul_hi ai bj
  let n_h = n_h + ck_h
  let c_h = S_bool ( (S_D (n_h >> lenS)) < (S_D (h >> lenS)) )
  let n_c = c + c_h
  in  (n_l, n_h, n_c)

def combineQ [Q] (accums: [Q][2]D) (carries: [Q]S) : [Q+2]D = #[unsafe]
  let lhcs = #[scratch] replicate (Q+2) zeroD
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
        let h_res = h + D_S (c_res + S_bool (lhcs[q] < l))
        let c_res = carries[q] + S_bool (h_res < h)
        in  (lhcs, h_res, c_res)
  let lhcs[Q]   = h_res
  let lhcs[Q+1] = D_S c_res
  in  lhcs

def convolutionQ [n] (Q: i64) (offset: i32) (k1: i32) (ash: [n]D) (bsh: [n]D) : [Q+2]D =
  #[unsafe]
  let accums = #[scratch] replicate Q (replicate 2 zeroD)
  let carries= #[scratch] replicate Q zeroS
  let Q = i32.i64 Q

  let (accums, carries) =
    loop (accums, carries) for q < Q do
      let accums[q,0] = zeroD
      let accums[q,1] = zeroD
      let carries[q]  = 0
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
def wrapperConvQ (Q: i64) (n: i32) (ash: []D) (bsh: []D) (tid: i32) : ([Q+2]D, [Q+2]D) =
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

def bmulRegsQ [IPB][M][Q] (Areg: [IPB*M][2*Q]D) (Breg: [IPB*M][2*Q]D) : [IPB*M][2*Q]D = 
  #[unsafe]
  let Ash = cpReg2Shm Areg -- this works in double Q
  let Bsh = cpReg2Shm Breg -- this works in double Q
  --
  let (vec_lhcs0, vec_lhcs1) =
    unzip <| opaque <|
    #[toregmem(1)] map (wrapperConvQ Q (i32.i64 M) Ash Bsh)
    <| map i32.i64 <| iota (IPB*M)
  --
  let (Lreg, Hreg) = 
    -- from4Reg2ShmQ vec_lhcs0 vec_lhcs1
    from4Reg2ShmQ Ash Bsh (copy vec_lhcs0) (copy vec_lhcs1)
  --
  let Rreg = badd0 Lreg Hreg
  in  Rreg

def bmul [ipb][n][q] (As: [ipb*n][2*q]D) (Bs: [ipb*n][2*q]D) : [ipb*n][2*q]D =
  #[unsafe]
  let Areg = #[glb2reg_only(1)] manifest As
  let Breg = #[glb2reg_only(1)] manifest Bs
  let Rreg = bmulRegsQ Areg Breg
  in  opaque Rreg

