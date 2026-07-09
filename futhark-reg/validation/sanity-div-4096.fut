import "divSanityTest"

-- ==
-- entry: bdiv4096Q4
-- "Bdiv4096Q4" script input { mkShinvInput 16384i64 1024i64 4i64 }
-- output { true }
entry bdiv4096Q4 [m] (uss0: [m][1024][4]u64) (vss0: [m][1024][4]u64) =
  #[unsafe]
  let uss = uss0 :> [m][1024][2*2]u64
  let vss = vss0 :> [m][1024][2*2]u64
  in  divTest uss vss
