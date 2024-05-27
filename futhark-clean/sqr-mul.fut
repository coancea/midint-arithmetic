import "big-add"

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

let combine2 (l0:D, h1:D, c2:S) (l1:D, h2:D, c3:S) : Dx4 =
  let l1' = l1 + h1
  let c2' = c2 + S_bool (l1' < l1) -- we assume carry is big enough to not overflow
  let h2' = h2 + D_S c2'
  let c3' = c3 + S_bool (h2' < h2)
  in  (l0, l1', h2', D_S c3')

let convolution4 (n: i32) 
                 (ash: []D) 
                 (bsh: []D)
                 (tid: i32) 
               : ( Dx4, i64x4, Dx4) =
 
  let instance = tid / n
  let vtid     = tid % n
  let offset = instance * (4*n)

  let computeIter64 (i: i32) (j: i32) (l: D, h: D, c: S) : (D, D, S) =
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

  -- first half:
  let k1 = 2*vtid
  let (lhc0, lhc1) =
    loop (lhc0, lhc1) = ((zeroD,zeroD,zeroS), (zeroD,zeroD,zeroS))
    for kk < k1 do
        let i = kk
        let j = k1 - i
        let lhc0 = computeIter64 i j lhc0
        let lhc1 = computeIter64 i (j+1) lhc1
        in  (lhc0, lhc1)
  let lhc1 = computeIter64 (k1+1) 0 lhc1
  let (l0, l1, h2, c3) = combine2 lhc0 lhc1
  let i0 = i64.i32 (offset + k1)
  
  -- second half
  let k2 = 4*n - k1 - 2
  let (lhc2, lhc3) =
    loop (lhc2, lhc3) = ((zeroD,zeroD,zeroS), (zeroD,zeroD,zeroS))
    for kk < k2+1 do
        let i = kk
        let j = k2 - i
        let lhc2 = computeIter64 i j lhc2
        let lhc3 = computeIter64 i (j+1) lhc3
        in  (lhc2, lhc3)
  let lhc3 = computeIter64 (k2+1) 0 lhc3
  let (l_nm2, l_nm1, h_n, c_np1) = combine2 lhc2 lhc3
  let i_nm2 = i64.i32 (offset + k2 - 1)
  in  ( (l0,     l1, l_nm2,   l_nm1  )
      , (i0,   i0+1, i_nm2,   i_nm2+1)
      , (h2,     c3, h_n,     c_np1  )
      )

  
let bmul [ipb][n] (as: [ipb*(4*n)]D) (bs : [ipb*(4*n)]D) : [ipb*(4*n)]D =
  #[unsafe]
  let nn = i32.i64 n
  let g = ipb * n  -- i64.i32 ((i32.i64 ipb) * nn)

  let cp2sh (i : i32) = #[unsafe]
        let g = i32.i64 g in
        ( ( as[i], as[g + i], as[2*g + i], as[3*g + i] )
        , ( bs[i], bs[g + i], bs[2*g + i], bs[3*g + i] ) )

  let ( ass, bss ) = iota g |> map i32.i64
                  |> map cp2sh  |> unzip
  let (a1s, a2s, a3s, a4s) = unzip4 ass
  let (b1s, b2s, b3s, b4s) = unzip4 bss
  let ash = a1s ++ a2s ++ a3s ++ a4s
  let bsh = b1s ++ b2s ++ b3s ++ b4s
  let ash = (ash :> [ipb*(4*n)]u64) |> opaque
  let bsh = (bsh :> [ipb*(4*n)]u64) |> opaque

  -- convolution
  let lsh = replicate (ipb*(4*n)) zeroD
  let hsh = replicate (ipb*(4*n)) zeroD
  let (ls, ils, hs) = iota g 
          |> map i32.i64
          |> map (convolution4 nn ash bsh)
          |> unzip3
  let ( l_0,  l_1, l_nm2, l_nm1) = unzip4 ls
  let (il_0, il_1,il_nm2,il_nm1) = unzip4 ils
  let ( h_0,  h_1, h_nm2, h_nm1) = unzip4 hs
  let lsh = scatter lsh (il_0++il_1++il_nm2++il_nm1)
                        (l_0 ++ l_1++ l_nm2++ l_nm1)
  let add2 (x: i64) = 2 + i32.i64 x |> i64.i32 
  let (ih_0, ih_1, ih_nm2, ih_nm1) =
        map (\ (a,b,c,d) -> (add2 a, add2 b, add2 c, add2 d) ) ils |> unzip4 
  let hsh = scatter hsh (ih_0++ih_1++ih_nm2++ih_nm1)
                        (h_0 ++ h_1++ h_nm2++ h_nm1)
  let rsh = badd0 ipb n lsh hsh
  in  rsh
  
  -- fake computation
--  let tupSum (l: D, h: D, c: S) : D = l + h + (D_S c)
--  let f (tid: i32) =
--        let (inst, vid) = ( tid / nn, tid % nn )
--        let (lhc0, lhc1, lhc2, lhc3) =
--            convolution4 nn ash bsh inst vid
--        in  (tupSum lhc0, tupSum lhc1, tupSum lhc2, tupSum lhc3)
--  let (fr0, fr1, fr2, fr3) = iota g |> map i32.i64 |> map f |> unzip4
--  let fr = fr0 ++ fr1 ++ fr2 ++ fr3
--  let frs = (fr :> [ipb*(4*n)]u64) |> opaque
--  in  frs
  


--
-- ==
-- entry: oneMultiply1024
-- compiled random input { [65536][1][1024]u64  [65536][1][1024]u64 }
entry oneMultiply1024 [m] (ass0: [m][1][1024]u64) (bss0: [m][1][1024]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][1*(4*256)]u64
   let bss = (map flatten bss0) :> [m][1*(4*256)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply512
-- compiled random input { [131072][1][512]u64  [131072][1][512]u64 }
entry oneMultiply512 [m] (ass0: [m][1][512]u64) (bss0: [m][1][512]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][1*(4*128)]u64
   let bss = (map flatten bss0) :> [m][1*(4*128)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply256
-- compiled random input { [262144][1][256]u64  [262144][1][256]u64 }
entry oneMultiply256 [m] (ass0: [m][1][256]u64) (bss0: [m][1][256]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][1*(4*64)]u64
   let bss = (map flatten bss0) :> [m][1*(4*64)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply128
-- compiled random input { [262144][2][128]u64  [262144][2][128]u64 }
entry oneMultiply128 [m] (ass0: [m][2][128]u64) (bss0: [m][2][128]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][2*(4*32)]u64
   let bss = (map flatten bss0) :> [m][2*(4*32)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply64
-- compiled random input { [262144][4][64]u64  [262144][4][64]u64 }
entry oneMultiply64 [m] (ass0: [m][4][64]u64) (bss0: [m][4][64]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][4*(4*16)]u64
   let bss = (map flatten bss0) :> [m][4*(4*16)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply32
-- compiled random input { [262144][8][32]u64  [262144][8][32]u64 }
entry oneMultiply32 [m] (ass0: [m][8][32]u64) (bss0: [m][8][32]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][8*(4*8)]u64
   let bss = (map flatten bss0) :> [m][8*(4*8)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss

--
-- ==
-- entry: oneMultiply16
-- compiled random input { [262144][16][16]u64  [262144][16][16]u64 }
entry oneMultiply16 [m] (ass0: [m][16][16]u64) (bss0: [m][16][16]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][16*(4*4)]u64
   let bss = (map flatten bss0) :> [m][16*(4*4)]u64
   let rss = imap2Intra ass bss bmul |> map unflatten
   in  rss


-- Big-Integer Multiplication: performance
-- ==
-- entry:  poly oneMul
-- compiled random input { 1024i64 [16384][1][4096]u64  [16384][1][4096]u64 }
-- compiled random input { 512i64 [32768][1][2048]u64  [32768][1][2048]u64 }
-- compiled random input { 256i64 [65536][1][1024]u64  [65536][1][1024]u64 }
-- compiled random input { 128i64 [131072][1][512]u64  [131072][1][512]u64 }
-- compiled random input { 64i64  [262144][1][256]u64  [262144][1][256]u64 }
-- compiled random input { 32i64  [262144][2][128]u64  [262144][2][128]u64 }
-- compiled random input { 16i64  [262144][4][64]u64   [262144][4][64]u64 }
-- compiled random input {  8i64  [262144][8][32]u64   [262144][8][32]u64 }
-- compiled random input {  4i64  [262144][16][16]u64  [262144][16][16]u64 }


-- compiled random input { [1024][4][64]u64  [1024][4][64]u64 }

-- computes one batched multiplication: a * b
entry oneMul [m][ipb] (n4: i64) 
                      (ass0: [m][ipb][4*n4]u64) 
                      (bss0: [m][ipb][4*n4]u64) 
                    : [m][ipb][4*n4]u64 = 
  #[unsafe]
  let ass = (map flatten ass0) 
  let bss = (map flatten bss0) 
  let rss = imap2Intra ass bss bmul |> map unflatten
  in  rss

-- computes: (a^2 + b) * (b^2 + b) + a*b
entry poly [m][ipb] (n4: i64) (ass0: [m][ipb][4*n4]u64) (bss0: [m][ipb][4*n4]u64) : [m][ipb][4*n4]u64 = 
  #[unsafe]
  let ass = (map flatten ass0) 
  let bss = (map flatten bss0) in
   imap2Intra ass bss
    (\ a b ->
        let a2   = bmul a  a            -- a^2
        let a2pb = badd0 ipb n4 a2 b    -- a^2 + b
        let b2   = bmul b  b            -- b^2
        let b2pb = badd0 ipb n4 b2 b    -- b^2 + b
        let prod = bmul a2pb b2pb       -- (a^2 + b) * (b^2 + b)
        let ab   = bmul a  b            -- a*b
        let res  = badd0 ipb n4 prod ab -- (a^2 + b) * (b^2 + b) + a*b
        in  res
    ) |> map unflatten

