import "../../intrinsics-accs"

type etp = u32

def mapIntra f as = #[incremental_flattening(only_intra)] map f as
def map3Intra f as bs cs = #[incremental_flattening(only_intra)] map3 f as bs cs

def getBits (bit_beg: u32) (num_bits: u32) (x: u32) : u32 =
  let mask = (1 << num_bits) - 1
  in (x >> bit_beg) & mask

def isBitUnset1 (bit_num: u32) (x: u32) : u32 =
  let shft = x >> bit_num
  in 1 - (shft & 1)

def isBitUnset (bit_num: u32) (x: u32) : bool =
  let shft = x >> bit_num
  in 0 == (shft & 1)

def B = 256i64
def Q =  22i64

def felmpad (n: i64) (xs: []u32) (blkidx: i64) (locidx: i64) : u32 =
  let ind = blkidx * (B*Q) + locidx
  in  if ind < n
      then #[unsafe]xs[ind]
      else u32.highest

def felm (_n: i64) (xs: []u32) (blkidx: i64) (locidx: i64) : u32 =
  #[unsafe] xs[blkidx * (B*Q) + locidx]

def ker1Blk (bit_beg: u32)
            (lgH: u32)
            (ixfn : i64 -> i64 -> u32)
            (blkidx: i64)
          : [B]u16 =
  let histo = replicate B 0u32
  let facc (histo: *acc ([B]u32)) tid : acc ([B]u32) =
      loop histo for q < Q do
        let ind = q*B + tid
        let bin = getBits bit_beg lgH (ixfn blkidx ind)
        in  write histo (i64.u32 bin) 1u32
  let histo =
    reduce_by_index_stream histo (+) 0u32 facc (iota B)
  in  map u16.u32 histo

def ker2Blk (bit_beg: u32)
            (lgH: u32)
            (ixfn  : i64 -> i64 -> u32)
            (blkidx: i64)
            (histo_loc: [B]u16)
            (histo_glb: [B]i64)
          : (*[B][Q]u32, [B][Q]i64) =
  let fcpy (shm: *acc ([B*Q]u32)) tid : acc ([B*Q]u32) =
    loop shm for q < Q do
      let ind = q*B+tid in write shm ind (ixfn blkidx ind)
  let size = B*Q
  let shm = (#[scratch]replicate size 0u32) :> [B*Q]u32
  let shm = opaque <| scatter_stream shm fcpy (iota B)
  --
  let gcpy tid = #[sequential]map (\q -> shm[Q*tid + q]) (iota Q)
  let elms = opaque <| #[toregmem(1)] map gcpy (iota B)
  --
  let elms =
    loop elms
    for i < i32.u32 lgH do
      let ff tid =
        loop s = 0u16 for q < Q do
          let zo = isBitUnset (bit_beg + u32.i32 i) (elms[tid,q])
          in  s + u16.bool zo
      --
      let tmp_buff = opaque <| map ff (iota B)
      let buff  = scan (+) 0u16 tmp_buff
      let split = buff[B-1]
      --
      -- load buff[tid-1] in registers
      let ldReg tid =
        if tid == 0 then 0 else buff[tid-1]
      let prevs = opaque <| #[toregmem(1)] map ldReg (iota B)
      --
      let gg (shm: *acc ([B*Q]u32)) tid : acc ([B*Q]u32) =
        let s = prevs[tid] in
        (loop (shm,s)  for q < Q do
           let elm = elms[tid, q]
           let zo  = isBitUnset (bit_beg + u32.i32 i) elm
           let s   = s + u16.bool zo
           let pos = if zo then s - 1 else (split + (u16.i64 tid * u16.i64 Q) + u16.i64 q) - s
           in  ( write shm (i64.u16 pos) elm, s ) ).0
      let size = B*Q
      let shm = (#[scratch]replicate size 0u32) :> [B*Q]u32
      let shm = opaque <| scatter_stream shm gg (iota B)
      --
      let freg tid =
        let f1 q = shm[q*B + tid]
        let f2 q = shm[tid*Q + q]
        in  if i == (i32.u32 lgH) - 1
            then #[sequential] map f1 (iota Q)
            else #[sequential] map f2 (iota Q)
      let elms = #[toregmem(1)] map freg (iota B)
      in  elms
  -- end repeated-partitioning LOOP
  -- compute the partial destination index from the histograms
  let hist_loc = manifest histo_loc
  let hist_loc_scan = scan (+) 0u16 hist_loc
  let hh tid = histo_glb[tid] - i64.u16 hist_loc_scan[tid]
  let histo  = map hh (iota B)
  -- compute the final index
  let finalInd tid =
    let finner q = 
      let elm = elms[tid, q]
      let bin = getBits bit_beg lgH elm
      let glb_offset = histo[i32.u32 bin]
      in  glb_offset + (q*B + tid)
    in  #[sequential] map finner (iota Q)
    -- data_keys_out[glb_pos] = elm;
  let fin_inds = #[toregmem(1)] map finalInd (iota B)
  in  (elms, fin_inds)

def radixIter [m]
      (bit_beg: u32)
      (dst:*[m * (B*Q)]u32)
      (indf : i64 -> i64 -> u32)
    : *[m * (B*Q)]u32 =
  #[unsafe]
  let lgH = 8u32
  -- let xs' = opaque <| unflatten xs
  let hist16 = mapIntra (ker1Blk bit_beg lgH indf) (iota m)
  let hist64 =
    transpose hist16
    |> manifest
    |> flatten
    |> map i64.u16
    |> scan (+) 0i64
  let hist64T =
    unflatten hist64
    |> transpose |> manifest
  -- let xs' = opaque <| unflatten xs
  let (xs'', inds') =
    unzip
    <| map3Intra (ker2Blk bit_beg lgH indf) (iota m) hist16 hist64T 
  in scatter dst (flatten (map flatten inds')) (flatten (map flatten xs''))

def radixSortU32 (n: i64) (ixfn : i64 -> i64 -> u32) : []u32 = -- *[m*(B*Q)]u32 =
  #[unsafe]
  let m = (n + (B*Q-1)) / (B*Q)
  let size = (m * (B*Q))
  let tmp = (#[scratch] replicate size 0u32) :> [m * (B*Q)]u32
  
  let xs' = radixIter 0 tmp ixfn
  
  let tmp1 = (#[scratch]replicate size 0u32) :> [m * (B*Q)]u32
  
  let (xs_res, _) =
    loop (xs', tmp1) for im1 < 3i32 do
      let i = im1 + 1
      let xs'' = radixIter (8 * u32.i32 i) tmp1 (felm size xs')
      in  (xs'', xs')
  in xs_res

-- ==
-- entry: mainU32
-- compiled random input { [100000000]u32 }

-- output { true } 
--

entry mainU32 [n] (xs: *[n]u32) =
  let xs' = radixSortU32 n (felmpad n xs)
  let success = 
        reduce (&&) true <|
        map (\ i -> xs'[i] <= xs'[i+1]) <|
        iota (n - 1)
  in xs'
--  in success

-- futhark dataset -b --i64-bounds=16384:16384 -g i64 -g [92274688]u32 | ./radix-sort-eff -e radixSortU32
-- futhark dataset -b --i64-bounds=4:4 -g i64 -g [22528]u32 | ./radix-sort-eff -e radixSortU32
-- futhark dataset -b --i64-bounds=4:4 -g i64 --u32-bounds=0:255 -g [22528]u32 --u32-bounds=0:0 -g [22528]u32 | ./radix-sort-eff --load-cuda=ker2.cu  -e firstIter
