import "divSanityTest"

-- ==
-- entry: bdiv8192Q8
-- "Bdiv8192Q8" script input { mkShinvInput 8192i64 1024i64 8i64 }
-- output { true }
entry bdiv8192Q8 [m] (uss0: [m][1024][8]u64) (vss0: [m][1024][8]u64) =
  #[unsafe]
  let uss = uss0 :> [m][1024][2*4]u64
  let vss = vss0 :> [m][1024][2*4]u64
  in  divTest uss vss
