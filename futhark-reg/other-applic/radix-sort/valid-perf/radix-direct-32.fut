import "../helpers"
import "../radix-sort"
import "lib/github.com/diku-dk/sorts/radix_sort"

----------------
----------------
--- Key: U32 ---
----------------
----------------

-- ==
-- entry: perfNewU32 perfOldU32
-- compiled input { 
--    [ 1923660277u32, 332013979u32,  316390370u32, 1029450092u32, 4103581731u32
--    , 3641318103u32, 719628295u32, 2301358507u32, 2448719612u32, 3304141370u32
--    ] }
-- output {
--    [  316390370u32,  332013979u32,  719628295u32, 1029450092u32, 1923660277u32
--    , 2301358507u32, 2448719612u32, 3304141370u32, 3641318103u32, 4103581731u32
--    ] }
-- compiled random input {   [1000000]u32 }
-- compiled random input {  [10000000]u32 }
-- compiled random input { [100000000]u32 }


entry perfNewU32 [n] (xs: *[n]u32) =
  radixSort u32.highest u32.highest 32 id id getBitsU32 xs

entry perfOldU32 [n] (xs: *[n]u32) =
  radix_sort u32.num_bits u32.get_bit xs

-- ==
-- entry: validNewOldU32
-- compiled input { [10u32, 9u32, 8u32, 7u32, 6u32, 5u32, 4u32, 3u32, 2u32, 1u32] }
-- output { true }
--
-- compiled input { 
--    [ 1923660277u32, 332013979u32,  316390370u32, 1029450092u32, 4103581731u32
--    , 3641318103u32, 719628295u32, 2301358507u32, 2448719612u32, 3304141370u32
--    ] }
-- output { true }
--
-- compiled random input { [30000]u32 }
-- output {true }
--
-- compiled random input {   [1000000]u32 }
-- output {true }
-- compiled random input {  [10000000]u32 }
-- output {true }
-- compiled random input { [100000000]u32 }
-- output {true }

entry validNewOldU32 [n] (xs: [n]u32) =
  let res_new = radixSort u32.highest u32.highest 32 id id getBitsU32 xs
  let res_old = radix_sort u32.num_bits u32.get_bit xs
  in  reduce (&&) true <| map2 (==) res_new res_old

