import "types"
import "utils"
import "badd"
import "bsub"
import "bmulWrap"
import "bmulFull"
import "bmul"

--
-- | Calculates ` B^h - v*w`
--
def powDiff [m][q] (h: i32) (l: i32)
                   (vs: [m][2*q]uint, precV: i32)
                   (ws: [m][2*q]uint, precW: i32)
                 : (u32, [m][2*q]uint) =
    let L = precW + precV - l + 1
    --
    let max_mul_size =
        if (precV == 0 || precW == 0 || L >= h)
        then precV + precW else i32.max 0i32 (i32.min L (precV + precW))
    --
    -- compute vs * ws
    let v_mul_w = bmulW ((max_mul_size)) vs ws -- : [m][2*q]uint in reg  -- trace
    --
--    let prec_mul = prec v_mul_w
--    let v_mul_w  = assert((trace v_mul_w[1])[3] >= zero_uint && trace max_mul_size > 0 && (trace ws[1])[0] >= 0) v_mul_w
    --
    let case_out = if precV == 0 || precW == 0 then 1 else if L >= h then 2 else 3
    let (sign, res) =
      match case_out
      case 1 ->    -- precV == 0 || precW == 0
        (1, mkPowBMul m (2*q) one_uint h) -- B^h
      case 2 ->    -- L >= h
        #[inform_pardim_only(1)] manifest
         ( if (ltBpow v_mul_w h)
           then (1, bsubFrBpowReg h v_mul_w) -- (1, B^h - v_mul_w)
           else (0, bsubOfBpowReg v_mul_w h) -- (0, v_mul_w - B^h)
         )
      case _ ->
        let P = v_mul_w in
        let x = if null P then 1i32
                else if zero_uint == getIndFromRegArr (L-1) P then 2i32
                else 3i32
        let (sign, res) = 
          match x
            case 1 -> (1, P)
            case 2 -> (0, P)
            case _ -> (1, bsubFrBpowReg L P)  -- (1, B^L - P)
        in  (sign, #[inform_pardim_only(1)] manifest res)
    --
    in  (sign, #[inform_pardim_only(1)] manifest res)

def step [m][q] (h: i32) (mm: i32) (l: i32) (vs: [m][2*q]uint, precV: i32) (ws: [m][2*q]uint) : [m][2*q]uint =
  let g = 2i32
  let precW = prec ws
  let (sign, xs) = powDiff (h-mm) (l-g) (vs, precV) (ws, precW)
  --
  -- redundant computation of precision; please optimize
  let precX = prec xs
  let ys = bmulW ((precW + precX)) ws xs   -- trace precision mult
  --
  let ws_sft = shift mm ws
  let res = 
    if ( sign == 1u32 )
    then let ys_sft = shift (2*mm - h) ys
         let res = baddReg' ws_sft ys_sft
         in  res
    else let isZero  = nullUpToInd (2*mm - h) ys
         let ys_sft  = shift (2*mm - h) ys
         let ys_sft' = if isZero then ys_sft else baddOne ys_sft
         let ys_sft' = #[inform_pardim_only(1)] manifest ys_sft'
         let res = bsubReg' ws_sft ys_sft'
         in  res
  in  #[inform_pardim_only(1)] manifest res

def refine3 [m][q] (h: i32) (k: i32) (l: i32) (vs: [m][2*q]uint) (w_high : uint, w_low: uint) : [m][2*q]uint =
  -- form a big number ws from w_high and w_low at positions 3 and 2
  let fm1 tid =
    let row = #[scratch] replicate (2*q) zero_uint in
    loop row for j < 2*q do
      let idx = i32.i64 (tid*(2*q) + j) in
      let v = if idx == 2 then w_low else if idx == 3 then w_high else zero_uint
      let row[j] = v in row
  let ws = opaque <| imapReg (iota m) fm1
  --
  let mkLoopCount (h: i32) (k: i32) : i32 =
      let nf = f32.ceil <| f32.log2 <| f32.i32 <| h - k - 1
      in  2 + i32.max 0 (i32.f32 nf)
  let n = mkLoopCount h k
  --
  let g = 2i32
  --
  let (_, ws') =
    loop (l, ws) for i < n do
      let mm = i32.min l (h - k + 1 - l)
      let s  = i32.max 0 (k - 2*l + 1 - g)
      let vs'= shift (-s) vs
      let precV' = i32.max (k+1 - s) 0i32
      --
      let ws = step (k + l + mm - s + g) mm l (vs', precV') ws
      --
      let (l, z) = if i < 2 then (l, mm) else (l + mm - 1, 1)   -- trace i
      in  (l, shift (-z) ws)
  --
  let qq = if h - k < 2 then h - k - 4 else -2
  in  shift qq ws'

-- | Key Function computing the whole-shifted inverse, i.e.,
--   It computes: `floor (B^h / vs)`
--   Assumes `uint` has size in bits >= 16, i.e., u8 is not supported
--
def shinv [m][q] (vs: [m][2*q]uint) (h: i32) : [m][2*q]uint =
  let k = (prec vs) - 1
  --
  let spec2 = gtBpowMul (k+1) vs one_uint h
  let spec3 = gtBpowMul (k+1) vs (highest_uint / 2) (h-1)
  let spec4 = eqBpowMul (k+1) vs one_uint k
  let x = if k == 0 || k == -1 
                        then 1i32 -- vs < B => B^h quo v (one digit div)
          else if spec2 then 2i32 -- vs > 1*B^h  => 0
          else if spec3 then 3i32 -- vs > (B/2) * B^{h-1} => 1
          else if spec4 then 4i32 -- vs == B^k => B^{h-k}
          else 5i32               -- general case
  let rs =
    match x
    case 1 -> quoPowB m (2*q) h (getIndFromRegArr 0 vs)
    case 2 -> bigZero m (2*q)
    case 3 -> bigOne m (2*q)
    case 4 -> mkPowBMul m (2*q) one_uint (h-k)
    case _ -> -- general treatment
      let v_low = getIndFromRegArr (k-1) vs
      let v_high= getIndFromRegArr k vs
      let (w_high, w_low) = BcubeQuoV v_low v_high
      in  refine3 h k 2 vs (w_high, w_low)
  in #[inform_pardim_only(1)] manifest rs

----------------
--- division ---
----------------

def bdivReg [m][q] (us_glb: [m][2*q]uint) (vs: [m][2*q]uint) : ([m][2*q]uint, [m][2*q]uint) =
  let h  = precGlb us_glb
  let ws = shinv vs h
  let us = #[glb2reg_only(1)] manifest us_glb
  let qs = bmulSftFullRegs (i64.i32 h) (us :> [1*m][2*q]uint) (ws :> [1*m][2*q]uint)
    
  let ms = bmulRegsQ (vs :> [1*m][2*q]uint) qs
  let (qs, ms) = (qs :> [m][2*q]uint, ms :> [m][2*q]uint)
  --
  let (ms, qs) = -- handles delta == -1
    match gt ms us   -- ms > us
    case false -> (ms, qs)
    case _ -> (bsubReg' ms vs, bsubOfBpowReg qs 0)
  --
  let ms = #[inform_pardim_only(1)] manifest ms
  let qs = #[inform_pardim_only(1)] manifest qs
  --
  let rs = bsubReg' us ms -- initial reminder
  --
  let (qs, rs) = -- handles delta == 1
    match gt vs rs -- vs > rs
      case true -> (qs, rs)
      case _ -> (baddOne qs, bsubReg' rs vs)
  --
  let qs = #[inform_pardim_only(1)] manifest qs
  let rs = #[inform_pardim_only(1)] manifest rs
  --
  in (qs, rs)           

def bdiv [m][q] (Us: [m][2*q]uint) (Vs: [m][2*q]uint) : ([m][2*q]uint, [m][2*q]uint) =
  -- let Ureg = #[glb2reg_only(1)] manifest Us
  let Vreg = #[glb2reg_only(1)] manifest Vs
  let (Qreg, Rreg) = bdivReg Us Vreg
  in  opaque (Qreg, Rreg)

def bdivBatch [insts][m][q]
              (uss_glb: [insts][m][2*q]uint) 
              (vss_glb: [insts][m][2*q]uint) 
            : ([insts][m][2*q]uint, [insts][m][2*q]uint) =
  -- step 1: distributed computation of the whole-shifted index
  let fshinv us_glb vs_glb =
    let us = #[glb2reg_only(1)] manifest us_glb
    let h  = prec us
    let vs = #[glb2reg_only(1)] manifest vs_glb
    in  (h, shinv vs h)
  let (hs_glb, wss_glb) = unzip <| opaque <| imap2Intra uss_glb vss_glb fshinv
  -- step 2: compute the full multiplication:
  let fFMul h us_glb ws_glb =
    let us = #[glb2reg_only(1)] manifest us_glb
    let ws = #[glb2reg_only(1)] manifest ws_glb
    in  bmulSftFullRegs (i64.i32 h) (us :> [1*m][2*q]uint) (ws :> [1*m][2*q]uint)
  let qss_glb = opaque <| imap3Intra hs_glb uss_glb wss_glb fFMul
  -- step 3: perform the rest of the computation
  let fMulSub us_glb vs_glb qs_glb =
    let us = #[glb2reg_only(1)] manifest us_glb
    let vs = #[glb2reg_only(1)] manifest vs_glb
    let qs = #[glb2reg_only(1)] manifest qs_glb
    --
    let ms = bmulRegsQ (vs :> [1*m][2*q]uint) qs
    let (qs, ms) = (qs :> [m][2*q]uint, ms :> [m][2*q]uint)
    --
    let (ms, qs) = -- handles delta == -1
      match gt ms us   -- ms > us
      case false -> (ms, qs)
      case _ -> (bsubReg' ms vs, bsubOfBpowReg qs 0)
    let ms = #[inform_pardim_only(1)] manifest ms
    let qs = #[inform_pardim_only(1)] manifest qs
    --
    let rs = bsubReg' us ms -- initial reminder
    --
    let (qs, rs) = -- handles delta == 1
      match gt vs rs -- vs > rs
        case true -> (qs, rs)
        case _ -> (baddOne qs, bsubReg' rs vs)
    let qs = #[inform_pardim_only(1)] manifest qs
    let rs = #[inform_pardim_only(1)] manifest rs
    in (qs, rs)
  --
  in unzip <| opaque <| imap3Intra uss_glb vss_glb qss_glb fMulSub

