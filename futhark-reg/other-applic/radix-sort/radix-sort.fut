import "../../intrinsics-accs"
import "helpers"

def B = 256i64
def Q =  25i64

def ker2Blk 't
      (dummy: t)
      (getBits: u32 -> u32 -> t -> u32)
      (bit_beg: u32)
      (lgH: u32)
      (ixfn  : i64 -> *t)
      (blkidx: i64)
      (histo_loc: [B]u16)
      (histo_glb: [B]i64)
    : (*[B][Q]t, [B][Q]i64) =
  let shm = glb2ShmIxf (copy dummy) B Q blkidx ixfn
  let elms = shm2Reg shm
  --
  let elms =
    loop elms
    for i < i32.u32 lgH do
      let buff = scanUnsetBits getBits (bit_beg + u32.i32 i) elms
      let split = buff[B-1]
      --
      -- load buff[tid-1] in registers
      let ldReg tid = if tid == 0 then 0 else buff[tid-1]
      let prevs = opaque <| #[toregmem(1)] map ldReg (iota B)
      --
      let gg (shm: *acc ([B*Q]t)) tid : acc ([B*Q]t) =
        let s = prevs[tid] in
        (loop (shm,s)  for q < Q do
           let elm = elms[tid, q]
           let zo  = isBitUnset getBits (bit_beg + u32.i32 i) elm
           let s   = s + u16.bool zo
           let pos = if zo then s - 1 else (split + (u16.i64 tid * u16.i64 Q) + u16.i64 q) - s
           in  ( write shm (i64.u16 pos) elm, s ) ).0
      let size = B*Q
      let shm = (#[scratch]replicate size (copy dummy)) :> [B*Q]t
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

def radixIter 't 'tu [m]
      (highest: tu)
      (from_bits: tu -> *t)
      (getBits: u32->u32->tu->u32)
      (bit_beg: u32)
      (dst:*[m * (B*Q)]t)
      (indf : i64 -> *tu)
    : *[m * (B*Q)]t =
  #[unsafe]
  let lgH = 8u32
  let hist16 = mapIntra (ker1Blk B Q getBits bit_beg lgH indf) (iota m)
  let hist64 = trScanTr hist16
  let (xs'', inds') =
    unzip <| map3Intra (ker2Blk highest getBits bit_beg lgH indf) (iota m) hist16 hist64
  let xs''' = map (map (map from_bits)) xs''
  in scatter dst (flatten (map flatten inds')) (flatten (map flatten xs'''))

def radixSort 't 'tu [n] (dummy: t) (highest: tu) (num_bits: i32) (to_bits: t -> *tu) (from_bits: tu -> *t) (getBits: u32->u32->tu->u32) (xs: [n]t) : [n]t =
-- def radixSortU32 (n: i64) (ixfn : i64 -> u32) : [n]u32 =
  #[unsafe]
  let m = (n + (B*Q-1)) / (B*Q)
  let size = (m * (B*Q))
  let tmp = (#[scratch] replicate size dummy) :> [m * (B*Q)]t
  
  let xs' = radixIter highest from_bits getBits 0 tmp (felmpad2uint to_bits highest xs)
  
  let tmp1 = (#[scratch]replicate size dummy) :> [m * (B*Q)]t
  
  let num_iters_min_1 = (num_bits + 7) / 8 - 1
  
  let (xs_res, _) =
    loop (xs', tmp1) for im1 < num_iters_min_1 do
      let i = im1 + 1
      let xs'' = radixIter highest from_bits getBits (8 * u32.i32 i) tmp1 (felm2uint to_bits xs')
      in  (xs'', xs')
  in xs_res[:n]

-- futhark dataset -b --i64-bounds=16384:16384 -g i64 -g [92274688]u32 | ./radix-sort-eff -e mainU32
-- futhark dataset -b --i64-bounds=4:4 -g i64 --u32-bounds=0:255 -g [22528]u32 | ./radix-sort-eff -e mainU32
