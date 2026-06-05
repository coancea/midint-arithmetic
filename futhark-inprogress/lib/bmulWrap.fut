import "../intrinsics-accs"
import "types"
import "badd"
import "bmul"

def combine2 (l0:uint, h1:uint, c2:uint_c) (l1:uint, h2:uint, c3:uint_c) : [4]uint =
  #[unsafe]
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

def bmul1 [m][q] (msz: i64)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  #[unsafe]
  let fm1 (tid: i64) =
      -- let (low, high, carry) = (zero_uint, zero_uint, zero_c) in
      -- if tid >= msz then (low, high, carry) else
      let (low, high, carry) =
        loop (low, high, carry) = (zero_uint, zero_uint, zero_c)
          for i < (tid+1) * i64.bool (tid < msz) do
            computeIter64 0 (i32.i64 i) (i32.i64 (tid - i)) Ash Bsh (low, high, carry)
      in ( #[sequential]replicate 1 low
         , #[sequential]replicate 1 high
         , #[sequential]replicate 1 carry )
  --
  let (lows, highs, carries) =
      unzip3 <| opaque <| #[toregmem(1)] map fm1 (iota m)
  --
  let fLacc (Lacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      if tid >= msz then Lacc else write Lacc tid lows[tid,0]
  let Lsh = opaque <| scatter_stream Ash fLacc (iota m)
  --
  let fHacc (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      if tid >= msz then Hacc else
      let Hacc = write Hacc tid highs[tid,0]
      in  write Hacc (tid+m) (uint_c carries[tid,0])
  let Hsh = opaque <| scatter_stream Bsh fHacc (iota m)
  --
  -- now load to 4 registers and form the lhcs vectors
  let fm2 tid =
      #[unsafe]
      let ind  = 2*tid
      let lhc1 =
          if ind < msz
          then (Lsh[ind], Hsh[ind], c_uint Hsh[m+ind])
          else (zero_uint, zero_uint, zero_c)
      let ind  = ind + 1
      let lhc2 =
          if ind < msz
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
      let ind  = 2*tid
      let Lacc = if ind < msz then write Lacc ind llhcs[tid,0] else Lacc
      let ind  = ind+1
      let Lacc = if ind < msz then write Lacc ind llhcs[tid,1] else Lacc
      in  Lacc
  let Lsh = opaque <| scatter_stream Lsh fL2acc (iota m)
  --
  let fH2acc (Hacc: *acc ([(1*m)*(2*q)]uint)) (tid: i64) : acc ([(1*m)*(2*q)]uint) =
      let ind  = 2*tid+1
      let Hacc = if ind < msz then write Hacc ind llhcs[tid,2] else Hacc
      let ind  = ind+1
      let Hacc = if ind < msz then write Hacc ind llhcs[tid,3] else Hacc
      in  Hacc
  let Hsh = opaque <| scatter_stream Hsh fH2acc (iota m)
  --
  -- finally load in registers according to original m and q
  let fm3 tid =
    #[unsafe]
    let Lreg = #[scratch] replicate (2*q) zero_uint
    let Hreg = #[scratch] replicate (2*q) zero_uint
    in
    loop (Lreg, Hreg) for i < 2*q do
      let ind = tid*(2*q) + i
      let Lreg[i] = if ind < msz then Lsh[ind] else zero_uint
      let Hreg[i] = if ind > 1 && ind < msz then Hsh[ind] else zero_uint
      in  (Lreg, Hreg)
  let (Lregs, Hregs) =
      unzip2 <| opaque <| #[toregmem(1)] map fm3 (iota (1*m))
  in  (Lregs, Hregs)

def bmulH [m][q] (msz: i64)
                 (qsz: i64)
                 (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  let Hini = replicate (1*2) zero_uint
  let (Lsh, Hsh, _) = bmulShmQ m qsz Hini Ash Bsh
  in  ( cpShm2RegPad msz zero_uint Lsh
      , cpShm2RegPad msz zero_uint Hsh )

def bmulF [m][q] (Ash: *[(1*m)*(2*q)]uint) 
                 (Bsh: *[(1*m)*(2*q)]uint)
               : ([1*m][2*q]uint, [1*m][2*q]uint) =
  let Hini = replicate (1*2) zero_uint
  let (Lsh, Hsh, _) = bmulShmQ m q Hini Ash Bsh
  in  (cpShm2Reg Lsh, cpShm2Reg Hsh)

-- Intends input arrays and result allocated in registers.
def bmulW [q][m] (maxSize: i64) (Areg0: [m][2*q]uint) (Breg0: [m][2*q]uint) : [m][2*q]uint =
  #[unsafe]
  let Areg = Areg0 :> [1*m][2*q]uint
  let Breg = Breg0 :> [1*m][2*q]uint
  let Ash = cpReg2Shm Areg
  let Bsh = cpReg2Shm Breg
  --
  let x = if maxSize <= m then 1
          else 
          if maxSize <= 2*m && q > 1 then 2
          else if maxSize <= 4*m && q > 2 then 3
          else 8
  let (Lreg, Hreg) =
    match x
    case 1 -> bmul1 maxSize Ash Bsh
    case 2 -> bmulH maxSize 1 Ash Bsh
    case 3 -> bmulH maxSize 2 Ash Bsh
    case _ -> bmulF Ash Bsh
  --
  -- #[inform_pardim_only(1)] manifest Lreg
  let Rreg = baddReg Lreg Hreg
  in  Rreg :> [m][2*q]uint
