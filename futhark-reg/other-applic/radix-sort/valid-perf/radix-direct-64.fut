import "../helpers"
import "../radix-sort"
import "lib/github.com/diku-dk/sorts/radix_sort"

----------------
----------------
--- Key: F64 ---
----------------
----------------

-- ==
-- entry: perfNewF64 perfOldF64
-- compiled input { 
--    [ 1923660277f64, 332013979f64,  316390370f64, 1029450092f64, 4103581731f64
--    , 3641318103f64, 719628295f64, 2301358507f64, 2448719612f64, 3304141370f64
--    ] }
-- output {
--    [  316390370f64,  332013979f64,  719628295f64, 1029450092f64, 1923660277f64
--    , 2301358507f64, 2448719612f64, 3304141370f64, 3641318103f64, 4103581731f64
--    ] }
-- compiled random input {   [1000000]f64 }
-- compiled random input {  [10000000]f64 }
-- compiled random input { [100000000]f64 }


entry perfNewF64 [n] (xs: *[n]f64) =
  radixSort 0f64 u64.highest 64 f64.to_bits f64.from_bits getBitsU64 xs

entry perfOldF64 [n] (xs: *[n]f64) =
  radix_sort_float f64.num_bits f64.get_bit xs

-- ==
-- entry: validNewOldF64
-- compiled input { [10f64, 9f64, 8f64, 7f64, 6f64, 5f64, 4f64, 3f64, 2f64, 1f64] }
-- output { true }
--
-- compiled input { 
--    [ 1923660277f64, 332013979f64,  316390370f64, 1029450092f64, 4103581731f64
--    , 3641318103f64, 719628295f64, 2301358507f64, 2448719612f64, 3304141370f64
--    ] }
-- output { true }
--
-- compiled random input { [30000]f64 }
-- output {true }
--
-- compiled random input {   [1000000]f64 }
-- output {true }
-- compiled random input {  [10000000]f64 }
-- output {true }
-- compiled random input { [100000000]f64 }
-- output {true }

entry validNewOldF64 [n] (xs: [n]f64) =
  let res_new = radixSort 0f64 u64.highest 64 f64.to_bits f64.from_bits getBitsU64 xs
  let res_old = radix_sort_float f64.num_bits f64.get_bit xs
  in  reduce (&&) true <| map2 (==) res_new res_old

