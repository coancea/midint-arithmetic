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

-----------------------------------------
--- Helper functions for division
-----------------------------------------

def prec [m][q] (vss : [m][q]uint) : i32 =
  #[unsafe]
  let ff i vs =
      loop p = 0i16 for j < q do
        if vs[j] == zero_uint
        then p
        else i16.i64 (i*q + j + 1)
  let hs = opaque <| map2 ff (iota m) vss
  in i32.i16 <| reduce_comm i16.max 0 hs -- can be replaced by hist

-- zero bigint array and set given index to d
def zeroAndSet (d : uint) (idx : i32) (m : i64) (q: i64) : [m][q]uint = 
  #[unsafe]
  opaque <|
  imapReg (iota m)
    (\i -> let rs = #[scratch] replicate q zero_uint in
           loop rs for j < q do
             let v = if idx == i32.i64 (i*q + j)
                     then d else zero_uint
             let rs[j] = v
             in  rs
    )

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
--  let ff vs = loop b = false for j < q do
--                if vs[j] != zero_uint then true else b
--  let hs = opaque <| map ff vss
--  in  ! (reduce_comm (||) false hs) -- can be replaced by hist

def modPow [m][q] (L: i32) (vss: [m][q]uint) : [m][q]uint =
  #[unsafe]
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

def bsubReg' [n][q] (areg : [n][q]uint) (breg : [n][q]uint) : [n][q]uint =
  #[unsafe]
  let res = bsubReg (areg :> [1*n][q]uint) (breg :> [1*n][q]uint)
  in  res :> [n][q]uint

def baddReg' [n][q] (areg : [n][q]uint) (breg : [n][q]uint) : [n][q]uint =
  #[unsafe]
  let res = baddReg (areg :> [1*n][q]uint) (breg :> [1*n][q]uint)
  in  res :> [n][q]uint

def shift [m][q] (n: i32) (xss: [m][q]uint) : [m][q]uint =
  #[unsafe]
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

