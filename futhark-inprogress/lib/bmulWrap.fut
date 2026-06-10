import "../intrinsics-accs"
import "types"
import "utils"
import "badd"
import "bmul"

def combine2 (l0:uint, h1:uint, c2:uint_c) (l1:uint, h2:uint, c3:uint_c) : [4]uint =
--  #[unsafe]
  let res = #[scratch] (#[sequential] replicate 4 zero_uint)
  let res[0] = l0
  let l1' = l1 + h1
  let res[1] = l1'
  -- let c2' = c2 + c_bool (l1' < l1) -- we assume carry is big enough to not overflow
  let c2' = c2 + c_bool (res[1] < l1)
  let h2' = h2 + uint_c c2'
  let res[2] = h2'
  let res[3] = uint_c (c3 + c_bool (res[2] < h2))
  in  res

-- | final load-up in registers according to original m and q
def finalRegLoad [m][q] (msz: i32)
                        (Lsh: *[(1*m)*(2*q)]uint) 
                        (Hsh: *[(1*m)*(2*q)]uint) =
  let fm3 tid =
    let Lreg = #[scratch] replicate (2*q) zero_uint
    let Hreg = #[scratch] replicate (2*q) zero_uint
    in
    loop (Lreg, Hreg) for i < 2*q do
      let ind = i32.i64 ( tid*(2*q) + i )
      let Lreg[i] = if ind < msz then Lsh[ind] else zero_uint
      let Hreg[i] = if ind > 1 && ind < msz then Hsh[ind] else zero_uint
      in  (Lreg, Hreg)
  --
  in  unzip2 <| opaque <| #[toregmem(1)] map fm3 (iota (1*m))

def bmul1 [m][q] (msz: i32)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
--  #[unsafe]
  let fm1 (tid: i64) =
      -- let (low, high, carry) = (zero_uint, zero_uint, zero_c) in
      -- if i32.i64 tid >= msz then (low, high, carry) else
      let (low, high, carry) =
        loop (low, high, carry) = (zero_uint, zero_uint, zero_c)
          for i < (tid+1) * i64.bool (i32.i64 tid < msz) do
            computeIter64 0 (i32.i64 i) (i32.i64 (tid - i)) Ash Bsh (low, high, carry)
      in ( #[sequential]replicate 1 low
         , #[sequential]replicate 1 high
         , #[sequential]replicate 1 carry )
  --
  let (lows, highs, carries) =
      unzip3 <| opaque <| #[toregmem(1)] map fm1 (iota m)
  --
  let fLacc (Lacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      if i32.i64 tid >= msz then Lacc else write Lacc tid lows[tid,0]
  let Lsh = opaque <| scatter_stream Ash fLacc (iota m)
  --
  let fHacc (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      if i32.i64 tid >= msz then Hacc else
      let Hacc = write Hacc tid highs[tid,0]
      in  write Hacc (tid+m) (uint_c carries[tid,0])
  let Hsh = opaque <| scatter_stream Bsh fHacc (iota m)
  --
  -- now load to 2*q registers and form the lhcs vectors
  let fm2 tid =
--      #[unsafe]
      let ind  = 2 * tid
      let lhc1 =
          if ind < i64.i32 msz
          then (Lsh[ind], Hsh[ind], c_uint Hsh[m+ind])
          else (zero_uint, zero_uint, zero_c)
      let ind  = ind + 1
      let lhc2 =
          if ind < i64.i32 msz
          then (Lsh[ind], Hsh[ind], c_uint Hsh[m+ind])
          else (zero_uint, zero_uint, zero_c)
      --
      let res = combine2 lhc1 lhc2
      in  res
  --
  let llhcs = opaque <| #[toregmem(1)] map fm2 (iota m)
  --
  -- now publish back to Lsh and Hsh in proper positions
  let fL2acc (Lacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let ind  = 2 * i32.i64 tid
      let Lacc = if ind < msz then write Lacc (i64.i32 ind) llhcs[tid,0] else Lacc
      let ind  = ind+1
      let Lacc = if ind < msz then write Lacc (i64.i32 ind) llhcs[tid,1] else Lacc
      in  Lacc
  let Lsh = opaque <| scatter_stream Lsh fL2acc (iota m)
  --
  let fH2acc (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let ind  = 2 * (i32.i64 tid) + 2
      let Hacc = if ind < msz then write Hacc (i64.i32 ind) llhcs[tid,2] else Hacc
      let ind  = ind+1
      let Hacc = if ind < msz then write Hacc (i64.i32 ind) llhcs[tid,3] else Hacc
      in  Hacc
  let Hsh = opaque <| scatter_stream Hsh fH2acc (iota m)
  --
  -- finally load in registers according to original m and q
  in  finalRegLoad msz Lsh Hsh

-----------------------------------
--- Size: 2*Q == 2, i.e., Q = 1
-----------------------------------

def bmulQ2 [m][q](msz: i32)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  let (vec_lhcs0, vec_lhcs1) = unzip <| opaque <|   -- ([m][Q+2]uint, [m][q+2]uint)
    #[toregmem(1)] map (wrapperConvQ 1i64 (i32.i64 m) Ash Bsh)
    <| map i32.i64 <| iota (1*m)
  let vec_lhcs0 = copy vec_lhcs0
  let vec_lhcs1 = copy vec_lhcs1
  --
  -- helper for writing an acc:
  let writeIf (myacc: *acc ([(1*m)*(2*q)]uint)) (offset: i64) (ind: i64) (v: uint) : *acc ([(1*m)*(2*q)]uint) =
      if i32.i64 ind < msz then write myacc (offset+ind) v else myacc
  -- place the results in shared memory
  --
  let fLacc (Lacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let Lacc = writeIf Lacc 0 tid vec_lhcs0[tid,0]
      in  writeIf Lacc 0 (2*m - tid - 1) vec_lhcs1[tid,0]
  let Lsh = opaque <| scatter_stream Ash fLacc (iota m)
  --
  let fHacc (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let Hacc = writeIf Hacc 0 tid vec_lhcs0[tid,1]
      let Hacc = writeIf Hacc (m*q) tid vec_lhcs0[tid,2]
      let ind  = 2*m - tid - 1
      let Hacc = writeIf Hacc 0 ind vec_lhcs1[tid,1]
      in  writeIf Hacc (m*q) ind vec_lhcs1[tid,2]
  let Hsh = opaque <| scatter_stream Bsh fHacc (iota m)
  --
  -- each thread gathers two consecutive entries
  let fm4 tid =
    let get3 idx =
        if idx < msz
        then (Lsh[idx], Hsh[idx], c_uint Hsh[idx + i32.i64 (m*q)])
        else (zero_uint, zero_uint, zero_c)
    let ind = i32.i64 (2*tid)
    let (lhc1, lhc2) = (get3 ind, get3 (ind+1))
    in  combine2 lhc1 lhc2
  let tup4s = opaque <| #[toregmem(1)] map fm4 (iota (1*m))
  --
  -- now publish again to Lsh and Hsh
  let fLacc4 (Lacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let Lacc = writeIf Lacc 0 (2 * tid) tup4s[tid,0]
      in  writeIf Lacc 0 (2*tid + 1) tup4s[tid,1]
  let Lsh = opaque <| scatter_stream Lsh fLacc4 (iota m)
  --
  let fHacc4 (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let Hacc = writeIf Hacc 0 (2*tid+2) tup4s[tid,2]
      in  writeIf Hacc 0 (2*tid+3) tup4s[tid,3]
  let Hsh = opaque <| scatter_stream Hsh fHacc4 (iota m)
  --
  -- finally load up to registers
  in  finalRegLoad msz Lsh Hsh

------------------------------
--- Half Q
------------------------------

def bmulH [m][q] (msz: i32)
                 (qsz: i64)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  let Hini = replicate (1*2) zero_uint
  let (Lsh, Hsh, _) = bmulShmQ m qsz Hini Ash Bsh
  in  ( cpShm2RegPad (i64.i32 msz) zero_uint Lsh
      , cpShm2RegPad (i64.i32 msz) zero_uint Hsh )

def bmulF [m][q] (msz: i32)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  let Hini = replicate (1*2) zero_uint
  let (Lsh, Hsh, _) = bmulShmQ m q Hini Ash Bsh
  in  ( cpShm2RegPad (i64.i32 msz) zero_uint Lsh
      , cpShm2RegPad (i64.i32 msz) zero_uint Hsh )

-- Intends input arrays and result allocated in registers.
def bmulW [q][m] (maxSize: i32) (Areg0: [m][2*q]uint) (Breg0: [m][2*q]uint) : [m][2*q]uint =
--  #[unsafe]
--  let maxSize = trace maxSize
  let Areg = Areg0 :> [1*m][2*q]uint
  let Breg = Breg0 :> [1*m][2*q]uint
  let Ash = cpReg2Shm Areg
  let Bsh = cpReg2Shm Breg
  --
  let x = if i64.i32 maxSize <= m then 1
          else if i64.i32 maxSize <= 2*m && q > 1 then 2
          else if i64.i32 maxSize <= 4*m && q > 2 then 3
          else 8
--
--  let x = if maxSize <= 2*m && q > 1 then 2
--          else if maxSize <= 4*m && q > 2 then 3
--          else 8

  let (Lreg, Hreg) =
    match x
    case 1 -> bmul1  maxSize Ash Bsh
    case 2 -> bmulQ2 maxSize Ash Bsh
    case 3 -> bmulH  maxSize 2 Ash Bsh
    case _ -> bmulF  maxSize Ash Bsh
  --
  let Lreg = #[inform_pardim_only(1)] manifest Lreg
  let Hreg = #[inform_pardim_only(1)] manifest Hreg
  let Rreg = baddReg Lreg Hreg
  in  Rreg :> [m][2*q]uint

