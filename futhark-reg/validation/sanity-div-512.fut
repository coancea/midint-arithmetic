import "divSanityTest"

-- ==
-- entry: bdiv512Q4
-- "Bdiv512Q4" script input { mkShinvInput 131072i64 128i64 4i64 }
-- output { true }
entry bdiv512Q4 [m] (uss0: [m][128][4]u64) (vss0: [m][128][4]u64)  =
  #[unsafe]
  let uss = uss0 :> [m][128][2*2]u64
  let vss = vss0 :> [m][128][2*2]u64
  in  divTest uss vss
