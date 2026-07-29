import "../../intrinsics-accs"
import "helpers"

def B = 256i64
def Q =  22i64

def ker2BlkKey 'tk 'tv
      (highest: tk)
      (dummyv: tv)
      (getBits: u32 -> u32 -> tk -> u32)
      (bit_beg: u32)
      (lgH: u32)
      (keyixf: i64 -> *tk)
      (datixf: i64 -> *tv)
      (blkidx: i64)
      (histo_loc: [B]u16)
      (histo_glb: [B]i64)
    : *([B][Q]tk, [B][Q]tv, [B][Q]i64) =
  -- copy from global to shared to register memory
  let shm = glb2ShmIxf highest B Q blkidx keyixf
  --
  let ftup tid q = let ind = Q*tid + q in (u16.i64 ind, shm[ind])
  let gcpy tid = unzip <| #[sequential]map (ftup tid) (iota Q)
  let (sigma, elms) = unzip <| opaque <| #[toregmem(1)] map gcpy (iota B)
  --
  let (sigma, elms) =
    loop (sigma, elms)
    for i < i32.u32 lgH do
      let buff = scanUnsetBits getBits (bit_beg + u32.i32 i) elms
      let split = buff[B-1]
      -- load buff[tid-1] in registers
      let ldRegPos tid =
        let s   = if tid == 0 then 0 else buff[tid-1]
        let pos = #[scratch] replicate Q 0u16 in
        (loop (s, pos) for q < Q do
            let elm = elms[tid, q]
            let zo  = isBitUnset getBits (bit_beg + u32.i32 i) elm
            let s   = s + u16.bool zo
            let ind = if zo then s - 1 else (split + (u16.i64 tid * u16.i64 Q) + u16.i64 q) - s
            let pos[q] = ind
            in (s, pos)
        ).1
      let poss = opaque <| #[toregmem(1)] map ldRegPos (iota B)
      --
      let gg 't (regs: [B][Q]t) (shm: *acc ([B*Q]t)) tid : acc ([B*Q]t) =
        loop shm for q < Q do write shm (i64.u16 poss[tid,q]) regs[tid,q]
      --
      let freg 't (shm: [B*Q]t) (tid: i64) : *[Q]t =
        let f1 q = shm[q*B + tid]
        let f2 q = shm[tid*Q + q]
        in  if i == (i32.u32 lgH) - 1
            then #[sequential] map f1 (iota Q)
            else #[sequential] map f2 (iota Q)
      --
      let size = B*Q
      let shm32 = (#[scratch]replicate size highest) :> [B*Q]tk
      let shm32 = opaque <| scatter_stream shm32 (gg elms) (iota B)
      --
      let elms = #[toregmem(1)] map (freg shm32) (iota B)
      -- 
      let shm16 = (#[scratch]replicate size 0u16) :> [B*Q]u16
      let shm16 = opaque <| scatter_stream shm16 (gg sigma) (iota B)
      --
      let fregN 't (shm: [B*Q]t) (tid: i64) : *[Q]t =
        #[sequential] map (\q -> shm[tid*Q + q]) (iota Q)
      let sigma = #[toregmem(1)] map (fregN shm16) (iota B)
      in  (sigma, elms)
  -- END repeated-partitioning LOOP
  -- copy the other data to shared memory:
  let shm = glb2ShmIxf dummyv B Q blkidx datixf
  -- load data permuted by sigma to registers 
  let sigmaPerm tid = #[sequential]map (\q -> shm[i32.u16 sigma[tid,q]]) (iota Q)
  let datelms = opaque <| #[toregmem(1)] map sigmaPerm (iota B)
  -- write them back to shared memory
  let cp2shm 't (regs: [B][Q]t) (shm: *acc ([B*Q]t)) tid : acc ([B*Q]t) =
        loop shm for q < Q do write shm (tid*Q+q) regs[tid,q]
  let shm = opaque <| scatter_stream shm (cp2shm datelms) (iota B)
  -- load back to regs but transposed
  let cpyShm2RegT tid = #[sequential]map (\q -> shm[q*B + tid]) (iota Q)
  let datelms = opaque <| #[toregmem(1)] map cpyShm2RegT (iota B)
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
  in  (elms, datelms, fin_inds)

def radixIterK 'tu 'tk 'tv [m]
      (highest: tu)
      (dummy: tv)
      (from_bits: tu -> *tk)
      (getBits: u32->u32->tu->u32)
      (bit_beg: u32)
      (dstkeysdata: *[m * (B*Q)](tk,tv))
      (keyixf : i64 -> *tu)
      (datixf : i64 -> *tv)
    : *[m * (B*Q)](tk,tv) =
  #[unsafe]
  let lgH = 8u32
  -- call first kernel that computes the histograms
  let hist16 = mapIntra (ker1Blk B Q getBits bit_beg lgH keyixf) (iota m)
  -- transpose, scan transpose back in i64
  let hist64 = trScanTr hist16
  -- call second kernel to produce
  -- (1) a per-block permutation
  -- (2) the per-block sorted keys
  -- (3) global indices for the result 
  let (keys, data, inds) =
    unzip3 <| map3Intra (ker2BlkKey highest dummy getBits bit_beg lgH keyixf datixf) (iota m) hist16 hist64
  let keys' = map (map (map from_bits)) keys
  in scatter dstkeysdata (flatten (map flatten inds)) (zip (flatten (map flatten keys')) (flatten (map flatten data)) )

def radixSortKey 'tu 'tk 'tv [n] (dummy: (tk,tv)) (highest: tu) (num_bits: i32) (to_bits: tk -> *tu) (from_bits: tu -> *tk) (getBits: u32->u32->tu->u32) (keys: [n]tk) (vals: [n]tv) : [n](tk,tv) =
  #[unsafe]
  let m = (n + (B*Q-1)) / (B*Q)
  let size = (m * (B*Q))
  let keydat_tmp = (#[scratch] replicate size dummy) :> *[m * (B*Q)](tk,tv)
  
  let xs' = radixIterK highest dummy.1 from_bits getBits 0 keydat_tmp (felmpad2uint to_bits highest keys) (felmpad dummy.1 vals)
  
  let keydat_tmp1 = (#[scratch] replicate size dummy) :> *[m * (B*Q)](tk,tv)
  
  let num_iters_min_1 = (num_bits + 7) / 8 - 1
  
  let (xs_res, _) =
    loop (xs', keydat_tmp1) for im1 < num_iters_min_1 do
      let i = im1 + 1
      let (keys', data') = unzip xs'
      let xs'' = radixIterK highest dummy.1 from_bits getBits (8 * u32.i32 i) keydat_tmp1 (felm2uint to_bits keys') (felm data')
      in  (xs'', xs')
  in xs_res[:n]

