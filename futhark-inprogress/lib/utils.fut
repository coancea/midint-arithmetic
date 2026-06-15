import "../intrinsics-accs"
import "types"
import "badd"
import "bsub"

def imapReg as f = #[toregmem(1)] map f as
def imap2Reg as bs f = #[toregmem(1)] map2 f as bs

-------------------------------------
--- fromReg2Shm and fromShm2Reg
-------------------------------------

def cpShm2Reg [M][Q] 't (Ash: [M*Q]t) : *[M][Q]t =
--  #[unsafe]
  let ff tid = #[sequential] map (\q -> Ash[tid*Q + q]) (iota Q)
  in  opaque <| #[toregmem(1)] map ff (iota M)

def cpShm2RegPad [M][Q] 't (m: i64) (pad: t) (Ash: [M*Q]t) : *[M][Q]t =
--  #[unsafe]
  let f tid j = let ind = tid*Q + j in if ind < m then Ash[ind] else pad
  let ff tid = #[sequential] map (f tid) (iota Q)
  in  opaque <| #[toregmem(1)] map ff (iota M)

def cpReg2Shm [M][Q] 't (Areg: [M][Q]t) : *[M*Q]t =
--  #[unsafe]
  let Ash = #[scratch] replicate (M*Q) Areg[0,0]
  let f (Aacc: *acc ([M*Q]t)) (tid: i64) : acc ([M*Q]t) =
    loop Aacc for q < Q do
      write Aacc (tid*Q + q) (Areg[tid, q])
  in opaque <| scatter_stream Ash f (iota M)

def cpReg2ShmNoAcc [M][Q] (Areg: [M][Q]uint) : *[M*Q]uint =
--  #[unsafe]
  let f (row : [Q]uint) =
    loop res = #[scratch] replicate Q (row[0]) for q < Q do
      let res[q] = row[q] in res
  in flatten (map f Areg)

-----------------------------------------
--- Helper functions for division
-----------------------------------------

def precRed [m][q] (vss : [m][q]uint) : i32 =
--  #[unsafe]
  let ff i vs =
      loop p = 0i16 for j < q do
        if vs[j] == zero_uint
        then p
        else i16.i64 (i*q + j + 1)
  let hs = opaque <| map2 ff (iota m) vss
  in i32.i16 <| reduce_comm i16.max 0 hs -- can be replaced by hist

def prec [m][q] (vss : [m][q]uint) : i32 =
--  #[unsafe]
  let shm = replicate 1 0i32
  let ffacc (myacc: *acc ([1]i32)) (tid: i64) : acc ([1]i32) =
      let ind =
        loop ind = 0i32 for i < q do
          if vss[tid, i] != zero_uint
          then i32.i64 (q * tid + i + 1) else ind
      in if ind > 0 then write myacc 0 ind else myacc
  --
  let shm = opaque <|
    reduce_by_index_stream shm (i32.max) 0i32 ffacc (iota m)
  in  shm[0]

def maxGtInd [m][q] (xss : [m][q]uint) (yss : [m][q]uint) : i32 =
  let shm = replicate 1 0i32
  let ffacc (myacc: *acc ([1]i32)) (tid: i64) : acc ([1]i32) =
      let ind =
        loop ind = 0i32 for i < q do
          if xss[tid, i] > yss[tid, i]
          then i32.i64 (q * tid + i + 1) else ind
      in if ind > 0 then write myacc 0 ind else myacc
  --
  let shm = opaque <|
    reduce_by_index_stream shm (i32.max) 0i32 ffacc (iota m)
  let max_index = shm[0]
  in  max_index

def gt [m][q] (xss : [m][q]uint) (yss : [m][q]uint) : bool =
  -- maxGtInd xss yss > maxGtInd yss xss
  let gtind = maxGtInd xss yss
  let shm = replicate 1 1i32
  let ffacc (myacc: *acc ([1]i32)) (tid: i64) : acc ([1]i32) =
      loop myacc for i < q do
        let ind = tid*q + i in
        if ind < i64.i32 gtind then myacc
        else if xss[tid,i] >= yss[tid,i] then myacc
        else write myacc 0 0i32
  let shm = opaque <| scatter_stream shm ffacc (iota m)
  in  shm[0] > 0
      
-- zero bigint array and set given index to d
def zeroAndSet (d : uint) (idx : i32) (m : i64) (q: i64) : [m][q]uint = 
--  #[unsafe]
  opaque <|
  imapReg (iota m)
    (\i -> let rs = #[scratch] replicate q zero_uint in
           loop rs for j < q do
             let v = if idx == i32.i64 (i*q + j)
                     then d else zero_uint
             let rs[j] = v
             in  rs
    )

-- | Answers: B^b > uss
def ltBpow [m][q] (uss: [m][q]u64) (b: i32) : bool =
  prec uss <= b

def nullUpToInd [m][q] (ind: i32) (xss: [m][q]uint) : bool =
  let resshm = replicate 1 true
  let ffacc (resacc: *acc ([1]bool)) (tid: i64) : acc ([1]bool) =
    let thd_res =
      loop thd_res = true for j < q do
        if (xss[tid,j] != zero_uint) && ( i32.i64 (q * tid + j) < ind )
        then false else thd_res
    in if thd_res then resacc else write resacc 0 false
  let resshm = opaque <| scatter_stream resshm ffacc (iota m)
  in resshm[0]  

def null [m][q] (vss: [m][q]uint) : bool =
  nullUpToInd (i32.i64 (m*q)) vss

def modPow [m][q] (L: i32) (vss: [m][q]uint) : [m][q]uint =
--  #[unsafe]
  opaque <|
  imap2Reg (iota m) vss
    (\ i vs ->
      let rs = #[scratch] replicate q 0 in
      loop rs for j < q do
        let rs[j] = if i32.i64 (i*q + j) < L then vs[j] else 0
        in  rs
    )

def getIndFromRegArr [m][q] (ind: i32) (arr: [m][q]uint) : uint =
  let tmpshm = replicate 1 zero_uint
  let fLacc (tmpacc: *acc ([1]uint)) (tid: i64) : acc ([1]uint) =
      loop tmpacc for j < q do
        if ind == i32.i64 (tid * q + j)
        then write tmpacc 0 arr[tid,j]
        else tmpacc
  let tmpshm = opaque <| scatter_stream tmpshm fLacc (iota m)
  in tmpshm[0]

def digitEqVal [m][q] (ind: i32) (v: uint) (arr: [m][q]uint) : bool =
  let tmpshm = replicate 1 false
  let fLacc (tmpacc: *acc ([1]bool)) (tid: i64) : acc ([1]bool) =
      loop tmpacc for j < q do
        if ind == i32.i64 (tid * q + j) && (arr[tid,j] == v)
        then write tmpacc 0 true
        else tmpacc
  let tmpshm = opaque <| scatter_stream tmpshm fLacc (iota m)
  in tmpshm[0]


def bsubReg' [n][q] (areg : [n][q]uint) (breg : [n][q]uint) : [n][q]uint =
--  #[unsafe]
  let res = bsubReg (areg :> [1*n][q]uint) (breg :> [1*n][q]uint)
  in  res :> [n][q]uint

def baddReg' [n][q] (areg : [n][q]uint) (breg : [n][q]uint) : [n][q]uint =
--  #[unsafe]
  let res = baddReg (areg :> [1*n][q]uint) (breg :> [1*n][q]uint)
  in  res :> [n][q]uint

def shift [m][q] (n: i32) (xss: [m][q]uint) : [m][q]uint =
--  #[unsafe]
  let shm = replicate (m*q) zero_uint
  let ffacc (tmpacc: *acc ([m*q]uint)) (tid: i64) : acc ([m*q]uint) =
      loop tmpacc for i < q do
        let idx = q * tid + i
        let offset = idx + i64.i32 n
        let (offset, value) =
            if offset >= 0 && offset < m*q
            then ( offset, xss[tid, i] )
            else ( m*q - idx - 1, zero_uint )
        in  write tmpacc offset value
  let shm = opaque <| scatter_stream shm ffacc (iota m)
  --
  in  cpShm2Reg shm


-- | Answers whether xs > a*B^n
--   Assumes that `a > 0` and that `prec` is the precision of `xs`
-- ToDo: optimize such that each thread performs one write to shm.
def gtBpowMul [m][q] (prec: i32) (xs: [m][q]uint) (a: uint) (n: i32) : bool =
--  #[unsafe]
  if prec > n+1 then true
  else if prec <= n then false
  else 
    let shm = replicate 3 0i32
    let ffacc (shmacc: *acc ([3]i32)) (tid: i64) : acc ([3]i32) =
      let (t0, t1, t2) =
        loop (t0,t1,t2)=(false,false,false)
        for i < q do
          let idx = i32.i64 (q * tid + i) in
          let v = xs[tid,i] in
          if v == zero_uint then (t0, t1, t2)
          else if idx < n
               then (true, t1,t2)
          else if idx == n && v == a
               then (t0, true, t2)
          else if idx == n && v > a
               then (t0, t1, true)
          else (t0, t1, t2)
      let shmacc = if t0 then write shmacc 0 1 else shmacc
      let shmacc = if t1 then write shmacc 1 1 else shmacc
      let shmacc = if t2 then write shmacc 2 1 else shmacc
      in  shmacc      
    let shm = opaque <| scatter_stream shm ffacc (iota m)
    in  if shm[2] > 0 then true
        else if shm[1] > 0 && shm[0] > 0 then true
        else false

-- | Answers: `a * B^{n} == xs`
--   Assumes `a > 0` and that `prec` is the precision of xs
-- ToDo: optimize such that each thread performs one write to shm
def eqBpowMul [m][q] (prec: i32) (xs: [m][q]uint) (a: uint) (n: i32) : bool =
--  #[unsafe]
  if prec < n+1 || prec > n+1 then false
  else 
  let shm = replicate 3 1i32
  let ffacc (shmacc: *acc ([3]i32)) (tid: i64) : acc ([3]i32) =
      let t0 =
        loop t0 = true for j < q do
          let v = xs[tid,j]
          let idx = i32.i64 (q * tid + j) in
          if (idx < n && v != 0) || (idx == n && v != a)
          then false else t0
      in  if !t0 then write shmacc 0 0 else shmacc
  let shm = opaque <| scatter_stream shm ffacc (iota m)
  in  shm[0] > 0

def bigZero (m: i64) (q: i64) : [m][q]uint =
--   #[unsafe]
   imapReg (iota m)
     (\ _ -> let row = #[scratch] replicate q zero_uint in
             loop row for j < q do let row[j] = zero_uint in row
     )


def bigZeroR (m: i64) (q: i64) : [m][q]uint =
--   #[unsafe]
   imapReg (iota m)
     (\ _ -> #[sequential] replicate q zero_uint )

def bigOne (m: i64) (q: i64) : [m][q]uint =
--   #[unsafe]
   imapReg (iota m)
     (\ i -> let z = #[sequential] replicate q zero_uint
             in if i != 0 then z
                else let z[0] = 1 in z                
     )

-- results in ` a * B^h `
def mkPowBMul (m: i64) (q: i64) (a: uint) (h: i32) : [m][q]uint =
--  #[unsafe]
  let f tid =
      let z = #[sequential] replicate q zero_uint
      let lb = tid*q
      in  if h >= i32.i64 lb && h < i32.i64 (lb+q)
          then let z[h - i32.i64 lb] = a in z
          else z
  in  opaque <| #[toregmem(1)] map f (iota m)

-- computes ` B^h quo d `, i.e., one digit division from a power of B
def quoPowB (m: i64) (q: i64) (h: i32) (d: uint) : [m][q]uint =
  if d == one_uint then mkPowBMul m q one_uint h
  else
    let shm = replicate (m*q) zero_uint
    let ffacc (shmacc: *acc ([m*q]uint)) (tid: i64) : acc ([m*q]uint) =
      if tid > 0 then shmacc else
      -- tid == 0 does all the computation:
      let r : uint128_t = { high = 0u64, low = 1u64 } in
      (loop (shmacc, r) for i_rev < h do
        let i = h - 1 - i_rev
        -- r = r << Base::bits
        let r : uint128_t = { high = r.low, low = 0u64 } -- r with high = r.low, low = 0u64
        in  if r.high > 0 || r.low >= d
            then let d128 : uint128_t = { high = 0u64, low = d } 
                 let (q, r) = divmod128 r d128
                 in  (write shmacc (i64.i32 i) q.low, r)
            else (shmacc, r)
      ).0
    let shm = opaque <| scatter_stream shm ffacc (iota m)
    in  cpShm2Reg shm

