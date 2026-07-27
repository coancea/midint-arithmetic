import "../../intrinsics-accs"

type etp = u32

def mapIntra f as = #[incremental_flattening(only_intra)] map f as
def map2Intra f as bs = #[incremental_flattening(only_intra)] map2 f as bs
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

def felmpad (n: i64) (xs: []u32) (ind: i64) : u32 =
  if ind < n then #[unsafe]xs[ind] else u32.highest

def felm 't (_n: i64) (xs: []t) (ind: i64) : t =
  #[unsafe] xs[ind]

def ker1Blk (bit_beg: u32)
            (lgH: u32)
            (ixfn : i64 -> u32)
            (blkidx: i64)
          : [B]u16 =
  let histo = replicate B 0u32
  let facc (histo: *acc ([B]u32)) tid : acc ([B]u32) =
      loop histo for q < Q do
        let ind = q*B + tid
        let elm = ixfn ( blkidx*(B*Q) + ind )
        let bin = getBits bit_beg lgH elm
        in  write histo (i64.u32 bin) 1u32
  let histo =
    reduce_by_index_stream histo (+) 0u32 facc (iota B)
  in  map u16.u32 histo

def cpGlb2Shm 't (blkidx: i64) (ixfn: i64 -> t)
      (shm: *acc ([B*Q]t)) (tid: i64) : acc ([B*Q]t) =
  loop shm for q < Q do
    let loc_ind = q*B+tid
    let glb_ind = blkidx*(B*Q) + loc_ind
    in  write shm loc_ind (ixfn glb_ind)

def ker2Blk 't
      (dummy: t)
      (bit_beg: u32)
      (lgH: u32)
      (keyixf: i64 -> u32)
      (datixf: i64 -> t)
      (blkidx: i64)
      (histo_loc: [B]u16)
      (histo_glb: [B]i64)
    : *([B][Q]u32, [B][Q]t, [B][Q]i64) =
  let size = B*Q
  -- copy from global to shared memory
  let shm = (#[scratch]replicate size 0u32) :> [B*Q]u32
  let shm = opaque <| scatter_stream shm (cpGlb2Shm blkidx keyixf) (iota B)
  --
  let gcpy tid = unzip <| #[sequential]map (\q -> let ind = Q*tid + q in (u16.i64 ind, shm[ind])) (iota Q)
  let (sigma, elms) = unzip <| opaque <| #[toregmem(1)] map gcpy (iota B)
  --
  let (sigma, elms) =
    loop (sigma, elms)
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
      let ldRegPos tid =
        let s   = if tid == 0 then 0 else buff[tid-1]
        let pos = #[scratch] replicate Q 0u16 in
        (loop (s, pos) for q < Q do
            let elm = elms[tid, q]
            let zo  = isBitUnset (bit_beg + u32.i32 i) elm
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
      let shm32 = (#[scratch]replicate size 0u32) :> [B*Q]u32
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
  -- copy the other data:
  let shm = (#[scratch]replicate size dummy) :> [B*Q]t
  let shm = opaque <| scatter_stream shm (cpGlb2Shm blkidx datixf) (iota B)
  --
  let cpyShm2Reg tid = #[sequential]map (\q -> shm[Q*tid + q]) (iota Q)
  let datelms = opaque <| #[toregmem(1)] map cpyShm2Reg (iota B)
  --
  let permshm 't (regs: [B][Q]t) (shm: *acc ([B*Q]t)) tid : acc ([B*Q]t) =
        loop shm for q < Q do write shm (i64.u16 sigma[tid,q]) regs[tid,q]
  let shm = (#[scratch]replicate size dummy) :> [B*Q]t
  let shm = opaque <| scatter_stream shm (permshm datelms) (iota B)
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

def radixIter [m] 't
      (dummy: t)
      (bit_beg: u32)
      (dstkeysdata: *[m * (B*Q)](u32,t))
      (keyixf : i64 -> u32)
      (datixf : i64 -> t)
    : *[m * (B*Q)](u32,t) =
  #[unsafe]
  let lgH = 8u32
  let hist16 = mapIntra (ker1Blk bit_beg lgH keyixf) (iota m)
  let hist64 =
    transpose hist16
    |> manifest
    |> flatten
    |> map i64.u16
    |> scan (+) 0i64
  let hist64T =
    unflatten hist64
    |> transpose |> manifest
  -- call kernel 2 to produce
  -- (1) a per-block permutation
  -- (2) the per-block sorted keys
  -- (3) global indices for the result 
  let (keys, data, inds) =  -- sigmas
    unzip3 <| map3Intra (ker2Blk dummy bit_beg lgH keyixf datixf) (iota m) hist16 hist64T
  in scatter dstkeysdata (flatten (map flatten inds)) (zip (flatten (map flatten keys)) (flatten (map flatten data)) )
  --
  -- do a per-block permutation of data
--  let blockPerm sigma blkidx =
--    let size = B*Q
--    let shm = (#[scratch]replicate size dummy) :> [B*Q]t
--    let shm = opaque <| scatter_stream shm (cpGlb2Shm blkidx datixf) (iota B)
--    --
--    let cpyShm2Reg tid = #[sequential]map (\q -> shm[Q*tid + q]) (iota Q)
--    let elms = opaque <| #[toregmem(1)] map cpyShm2Reg (iota B)
--    --
--    let permshm 't (regs: [B][Q]t) (shm: *acc ([B*Q]t)) tid : acc ([B*Q]t) =
--        loop shm for q < Q do write shm (i64.u16 sigma[tid,q]) regs[tid,q]
--    let shm = (#[scratch]replicate size dummy) :> [B*Q]t
--    let shm = opaque <| scatter_stream shm (permshm elms) (iota B)
--    -- load back to regs but transposed
--    let cpyShm2RegT tid = #[sequential]map (\q -> shm[q*B + tid]) (iota Q)
--    in  opaque <| #[toregmem(1)] map cpyShm2RegT (iota B)
--  let data = map2Intra blockPerm sigmas (iota m)
--  --
--  in scatter dstkeysdata (flatten (map flatten inds)) (zip (flatten (map flatten keys)) (flatten (map flatten data)) )
--  let (dstkeys, dstdata) = unzip dstkeysdata
--  let keys' = scatter dstkeys (flatten (map flatten inds1')) (flatten (map flatten keys))
--  let data' = scatter dstdata (flatten (map flatten inds2')) (flatten (map flatten data))
--  in  zip keys' data'

def radixSortU32 't (dummy: t) (n: i64) (keyixf : i64 -> u32) (datixf : i64 -> t) : [](u32,t) = -- *[m*(B*Q)]u32 =
  #[unsafe]
  let m = (n + (B*Q-1)) / (B*Q)
  let size = (m * (B*Q))
  let keydat_tmp = (#[scratch] replicate size (0u32,dummy)) :> *[m * (B*Q)](u32,t)
--  let dat_tmp = (#[scratch] replicate size dummy) :> [m * (B*Q)]t
    
  let xs' = radixIter dummy 0 keydat_tmp keyixf datixf
  
  let keydat_tmp1 = (#[scratch] replicate size (0u32,dummy)) :> *[m * (B*Q)](u32,t)
  -- let dat_tmp1 = (#[scratch] replicate size dummy) :> [m * (B*Q)]t
  
  let (xs_res, _) =
    loop (xs', keydat_tmp1) for im1 < 3i32 do
      let i = im1 + 1
      let (keys, data) = unzip xs'
      let xs'' = radixIter dummy (8 * u32.i32 i) keydat_tmp1 (felm size keys) (felm size data) 
      in  (xs'', xs')
  in xs_res

-- ==
-- entry: main
-- compiled random input { [100000000]u32 }

-- output { true } 
--

entry main [n] (xs: *[n]u32) =
  let (xs', iot') = unzip <| radixSortU32 0i32 n (felmpad n xs) (\ (ind : i64) -> i32.i64 ind )
  let success = 
        reduce (&&) true <|
        map (\ i -> xs'[i] <= xs'[i+1]) <|
        iota (n - 1)
--  in xs'
  in (xs', iot')
--  in success


-- futhark dataset -b --i64-bounds=16384:16384 -g i64 -g [92274688]u32 | ./radix-sort-eff -e mainU32
-- futhark dataset -b --i64-bounds=4:4 -g i64 --u32-bounds=0:255 -g [22528]u32 | ./radix-sort-eff -e mainU32
