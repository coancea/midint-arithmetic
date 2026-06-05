import "types"
import "utils"
import "badd"
import "bmulWrap"

--
-- Calculates B^h-v*w
-- ToDos:
--   1. if we support subtraction with B^{pow}, then not
--      only we peedup computation but also we save
--      registers, as we do not have to manifest B^{pow}
--
def powDiff [m][q] (h: i32) (l: i32) (vs: [m][2*q]uint) (ws: [m][2*q]uint) : (u32, [m][2*q]uint) =
    let precV = prec vs
    let precW = prec ws
    let L = precW + precV - l + 1

    let (powofb, max_mul_size) =
        if (precV == 0 || precW == 0 || L >= h)
        then (h, precV + precW) 
        else (L, L)
    --
    -- compute B^powofb
    let bpow = zeroAndSet one_uint powofb m (2*q)   -- : [m][2*q]uint in reg
    --
    -- compute vs * ws
    let v_mul_w = bmulW (i64.i32 max_mul_size) vs ws -- : [m][2*q]uint in reg
    -- let ret = convMulV2 vs ws
    --
    let case_out = if precV == 0 || precW == 0 then 1 else if L >= h then 2 else 3
    in
    match case_out
    case 1 ->    -- precV == 0 || precW == 0
      (1, bpow)   
    case 2 ->    -- L >= h
      if ltBpow v_mul_w h
      then (1, bsubReg' bpow v_mul_w)
      else (0, bsubReg' v_mul_w bpow)
    case _ ->
      let P = v_mul_w
      let is_zero_P = null P
      let P_Lm1 = getIndFromRegArr (L-1) P
      let (need_sub, sign) =
          if is_zero_P
          then (false, 1u32)
          else if P_Lm1 == zero_uint
               then (false, 0u32)
               else (true,  1u32)
      let P' = if !need_sub then P
               else bsubReg' bpow P
      in  (sign, P')

def step [m][q] (h: i32) (mm: i32) (l: i32) (vs: [m][2*q]uint) (ws: [m][2*q]uint) : [m][2*q]uint =
  let g = 2i32
  let (sign, xs) = powDiff (h-mm) (l-g) vs ws
  --
  -- redundant computation of precision; please optimize
  let precW = prec ws
  let precX = prec xs
  let ys = bmulW (i64.i32 (precW + precX)) ws xs
  --
  let ws_sft = shift mm ws in
  if sign == 1u32
  then let ys_sft = shift (2*mm - h) ys
       in  baddReg' ws_sft ys_sft
  else let isZero = nullUpToInd (2*mm - h) ys
       let ys_sft = shift (2*mm - h) ys
       let ys_sft'= if isZero then ys_sft else baddOne ys_sft
       in  bsubReg' ws_sft ys_sft'

