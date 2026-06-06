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
--   2. do not compute bpow ahead of time; it eats up regs
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

def refine3 [m][q] (h: i32) (k: i32) (l: i32) (vs: [m][q]uint) (ws: [m][q]uint) : [m][q]uint =
  #[unsafe]
  let mkLoopCount (h: i32) (k: i32) : i32 =
      let nf = f32.ceil <| f32.log2 <| h - k - 1
      in  2 + i32.max 0 (i32.f32 nf)
  --
  let g = 2i32
  let ws = shift g ws
  let n = mkLoopCount h k
  let ws' =
    loop (l, ws) for i < n do
      let mm = i32.min l (h - k + 1 - l)
      let s  = i32.max 0 (k - 2*l + 1 - g)
      let vs'= shift (-s) vs
      --
      let ws = step (k + l + mm - s + g) mm l vs' ws
      --
      let (l, z) = if i < 2 then (l, mm) else (l + mm - 1, 1)
      in  (l, shift (-z) ws)
  --
  let qq = if h - k < 2 then h - k - 4 else -2
  in  shift qq ws'

B^{k-1} <= v < B^{k} => prec vs = k
if v < B then return B^h quo v ▷ Divide by 1 digit
if v > B^h then return 0       ▷ get digit h   (precision h+1)
if 2v > B^h then return 1      ▷ get digit h-1 (precision h)  -- case B^h >= v > B^h / 2
if v = B^{k} then return B^{h-k} ▷ get digit k-1 (precision k+1)

def shinv [m][q] (vs: [m][q]uint) (h: i32) : [m][q]uint =
  -- ASSUMES uint has bit size >= 16, i.e., u8 is not supported
  let k = (prec vs) - 1
  --
  let (is_special, res) =
    if k == 0 then (true, quoPowB m q h (getIndFromRegArr 0 vs))  -- vs < B    => B^h quo v (one digit div)
    else if k > h || (k == h && one_uint < getIndFromRegArr h vs) -- vs > B^h  => zero
         then (true, zero m q)
    else if k == h || (k == h-1 && highest_uint / 2 < getIndFromRegArr (h-1) vs) -- B^h >= vs > B^h / 2 => 1
         then (true, one m q)
    else if v == B^k then (true, B^{h-k})
    else (false, zero m q)
  
  if is_special then res else
  -- general treatment
  ... to be continued ...

