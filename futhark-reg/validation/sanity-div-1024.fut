import "divSanityTest"

-- ==
-- entry: bdiv1024Q8
-- "Bdiv1024Q8" script input { mkShinvInput 65536i64 128i64 8i64 }
-- output { true }
entry bdiv1024Q8 [m] (uss0: [m][128][8]u64) (vss0: [m][128][8]u64) =
  #[unsafe]
  let uss = uss0 :> [m][128][2*4]u64
  let vss = vss0 :> [m][128][2*4]u64
  in  divTest uss vss

