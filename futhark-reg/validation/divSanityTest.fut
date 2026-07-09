----------------------------------------------------
--- This sanity check tests that if `(quo, rem)` ---
---   is the result of dividing `uss` with `vss` ---
---   then it holds that:                        ---
---       uss = quo * vss + rem                  ---
----------------------------------------------------

import "../performance/lib/github.com/diku-dk/cpprandom/random"
import "../lib/types"
import "../lib/badd"
import "../lib/bmul"
import "../lib/bdiv"

def divTest [n][m][q] (uss: [n][m][2*q]u64) (vss: [n][m][2*q]u64) : bool =
   let (quo, rem) = unzip <| opaque <| imap2Intra uss vss bdiv
   let tmp = opaque <| imap2Intra (vss :> [n][1*m][2*q]u64) (quo :> [n][1*m][2*q]u64) bmul
   let uss'= opaque <| imap2Intra tmp (rem  :> [n][1*m][2*q]u64) badd
   let uss_flat = flatten (map flatten (uss :> [n][1*m][2*q]u64))
   let uss_flat'= flatten (map flatten uss')
   in  uss_flat == uss_flat' 

-------------------------------------------
--- Utilities for constructing datasets ---
-------------------------------------------

def mkRandBIntFull (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64 in
  loop (rng_state, arr) for i < max_prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

def mkRandBIntPart (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64
  let (rng_state, r) = minstd_rand.rand rng_state
  let prec = ( (i32.u32 r) % (max_prec/2) ) + 2 in
  loop (rng_state, arr) for i < prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

---- ==
-- entry: mkShinvInput
-- compiled input { 32768i64 256i64 8i64 }
entry mkShinvInput (num_instances: i64) (m: i64) (q2: i64)
                 : ( [num_instances][m][q2]u64
                   , [num_instances][m][q2]u64
                   ) =
  let rng = minstd_rand.rng_from_seed [1i32]
  let rngs= minstd_rand.split_rng num_instances rng
  let max_prec   = i32.i64 <| m*q2 - 2
  let (rngs', uss) = unzip <| map (mkRandBIntFull max_prec m q2) rngs
  let (_,     vss) = unzip <| map (mkRandBIntPart max_prec m q2) rngs'
  in  (map unflatten uss, map unflatten vss)  -- replicate num_instances max_prec
  


