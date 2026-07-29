import "../../intrinsics-accs"

def mapIntra f as = #[incremental_flattening(only_intra)] map f as
def map2Intra f as bs = #[incremental_flattening(only_intra)] map2 f as bs
def map3Intra f as bs cs = #[incremental_flattening(only_intra)] map3 f as bs cs

def felmpad2uint 't 'tu [n] (to_bits: t -> *tu) (highest: tu) (xs: [n]t) (ind: i64) : *tu =
  if ind < n then to_bits (#[unsafe]xs[ind]) else copy highest

def felmpad 't [n] (highest: t) (xs: [n]t) (ind: i64) : *t =
  if ind < n then copy (#[unsafe]xs[ind]) else copy highest

def felm2uint 't 'tu (to_bits: t -> *tu) (xs: []t) (ind: i64) : *tu =
  to_bits (#[unsafe] xs[ind])

def felm 't  (xs: []t) (ind: i64) : *t =
  copy (#[unsafe] xs[ind])

def getBitsU32 (bit_beg: u32) (num_bits: u32) (x: u32) : u32 =
  let mask = (1 << num_bits) - 1
  in (x >> bit_beg) & mask

def isBitUnset1 (bit_num: u32) (x: u32) : u32 =
  let shft = x >> bit_num
  in 1 - (shft & 1)

--def isBitUnset 'tu (bit_num: u32) (x: tu) : bool =
--  let shft = x >> bit_num
--  in 0 == (shft & 1)
 
def isBitUnset 'tu (getBits: u32->u32->tu->u32) (bit_idx: u32) (x: tu) : bool =
  0 == getBits bit_idx 1 x
 
def trScanTr [m][B] (hist16: [m][B]u16) : [m][B]i64 =
  let hist64 =
    transpose hist16
    |> manifest
    |> flatten
    |> map i64.u16
    |> scan (+) 0i64
  in  unflatten hist64
    |> transpose |> manifest

def ker1Blk 't
      (B: i64)
      (Q: i64)
      (getBits: u32->u32->t->u32)
      (bit_beg: u32)
      (lgH: u32)
      (ixfn : i64 -> t)
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

-----------------------------------------------------
--- Helpers for Kernel 2 of Radix Sort
-----------------------------------------------------

def cpGlb2Shm 't [B][Q] (blkidx: i64) (ixfn: i64 -> t)
      (shm: *acc ([B*Q]t)) (tid: i64) : acc ([B*Q]t) =
  loop shm for q < Q do
    let loc_ind = q*B+tid
    let glb_ind = blkidx*(B*Q) + loc_ind
    in  write shm loc_ind (ixfn glb_ind)

def glb2ShmIxf 't (dummy: t) (B: i64) (Q: i64) (blkidx: i64) (ixfn: i64 -> *t) : *[B*Q]t =
  let fcpy (shm: *acc ([B*Q]t)) (tid: i64) : acc ([B*Q]t) =
    loop shm for q < Q do
      let loc_ind = q*B+tid
      let glb_ind = blkidx*(B*Q) + loc_ind
      in  write shm loc_ind (ixfn glb_ind)
  let size = B*Q
  let shm = (#[scratch]replicate size dummy) :> [B*Q]t
  in  opaque <| scatter_stream shm fcpy (iota B)

def shm2Reg 't [B] [Q] (shm: [B*Q]t) : [B][Q]t =
  let gcpy tid = #[sequential]map (\q -> shm[Q*tid + q]) (iota Q)
  in  opaque <| #[toregmem(1)] map gcpy (iota B)

def scanUnsetBits 't [B][Q] getBits (bit_idx: u32) (keys: [B][Q]t) : [B]u16 =
  let ff tid =
    loop s = 0u16 for q < Q do
      let zo = isBitUnset getBits bit_idx (keys[tid,q])
      in  s + u16.bool zo
  --
  let tmp_buff = opaque <| map ff (iota B)
  in  scan (+) 0u16 tmp_buff

