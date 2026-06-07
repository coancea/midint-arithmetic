import "../intrinsics-accs"

-----------------------------------------------------------
--- Types for the type of scan operator in big addition ---
-----------------------------------------------------------

type cT         = u32      --u8
let  cTfromBool = u32.bool --u8.bool
let  carryOpNE: cT = 2u32   --2u8

--type cT         = u8
--let  cTfromBool = u8.bool
--let  two_cT     = 2u8

------------------------------------
--- Types for big multiplication ---
------------------------------------

type uint  = u64
let uint_mul_hi = u64.mul_hi
let uint_bool = u64.bool
let zero_uint = 0u64
let one_uint  = 1u64
let highest_uint = u64.highest
let uint_cT   = u64.u32

--type D = u64
--let D_mul_hi = u64.mul_hi
--let D_bool = u64.bool
--let zeroD = 0u64

type uint_c = u32
let zero_c = 0u32
let size_c = 32u64
let c_bool = u32.bool
let c_uint = u32.u64
let uint_c = u64.u32

--type S = u32
--let zeroS = 0u32
--let lenS = 32u64
--let S_bool = u32.bool
--let S_D = u32.u64
--let D_S = u64.u32
--type Dx4   = (D,D,D,D)

type uint128_t = {high: u64, low: u64}

-- subtraction of uint128_t
def sub128 (a: uint128_t) (b: uint128_t) : uint128_t =
  let res_low = a.low - b.low
  let borrow  = u64.bool (res_low > a.low)
  let res_high= a.high - b.high - borrow
  in  { high = res_high, low = res_low }


-- | Performs 128-bit unsigned division and modulo operations.
--   dividend: the 128-bit value to be divided
--   divisor:  the 128-bit value to divide by
--   result: (quotient, reminder)
let divmod128 (dividend: uint128_t) (divisor: uint128_t) : (uint128_t, uint128_t) =
  #[unsafe]
  let q : uint128_t = { high = 0u64, low = 0u64 }
  let r : uint128_t = { high = 0u64, low = 0u64 } in
  loop (q,r) for i_rev < 128i32 do
    let i = 127 - i_rev
    -- Left shift remainder by 1 bit
    let r = r with high = (r.high << 1) | (r.low >> 63)
    let r = r with low  = r.low << 1
    -- Pull the i-th bit of the dividend into the least significant bit of remainder
    let bit =
        if i >= 64
        then (dividend.high >> u64.i32 (i - 64)) & 1
        else (dividend.low >> u64.i32 i) & 1
    let r = r with low = r.low | bit
    -- Compare remainder (r) with divisor
    let r_gte_divisor =
        if (r.high > divisor.high)
        then true
        else if (r.high == divisor.high)
             then if (r.low >= divisor.low)
                  then true
                  else false
             else false
    -- If remainder is greater than or equal to the divisor
    in if !r_gte_divisor
       then (q, r)
       else -- Subtract divisor from remainder
            let r = if (r.low < divisor.low) 
                    then r with high = r.high - 1 -- Handle borrow
                    else r
            let r = r with low = r.low - divisor.low
            let r = r with high= r.high- divisor.high
            -- Set the i-th bit of the quotient
            let q =
                if (i >= 64)
                then q with high = q.high | (1u64 << u64.i32 (i - 64))
                else q with low  = q.low  | (1u64 << u64.i32 i)
            in  (q, r)

def shft1_u128 (rem: uint128_t) : uint128_t =
  let r = rem.high << 1
  let r = r | u64.bool ((rem.low & u64.bool (1u64 < 63)) != 0)
  let rem = rem with high = r
  let rem = rem with low = rem.low << 1 
  in  rem

def BcubeQuoV vlow vhigh =
  let v : uint128_t = { high = vhigh, low = vlow }
  let quotient : uint128_t = { high = 0u64, low = 0u64 }
  let rem : uint128_t = { high = 0u64, low = 0u64 }
  let overflow = false in
  let (q, _, _) =
   loop (quotient, rem, overflow)
    for i_rev < 193 do
      let i = 192 - i_rev
      let overflow = if rem.high & (1u64 << 63) != 0 then true else overflow
      -- rem = rem << 1
      let rem = shft1_u128 rem
      let rem = if i != 192 then rem
                else rem with low = rem.low | 1
      let quotient = shft1_u128 quotient
      let geq a b =
        a.high > b.high || (a.high == b.high && a.low >= b.low)
      in  if (geq rem v) || overflow
          then ( quotient with low = quotient.low | 1
               , sub128 rem v
               , false
               )
          else (quotient, rem, overflow)
  in (q.high, q.low)

---------------------------------
--- SOAC renaming
---------------------------------

let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap3 as bs cs f = map3 f as bs cs

let imapIntra as f = #[incremental_flattening(only_intra)] map f as
let imap2Intra as bs f = #[incremental_flattening(only_intra)] map2 f as bs
let imap3Intra as bs cs f = #[incremental_flattening(only_intra)] map3 f as bs cs

