import "divSanityTest"

entry mkShinvInput = mkShinvInput0

-- ==
-- entry: bdiv128Q4
-- "Bdiv128Q4" script input { mkShinvInput 524288i64 32i64 4i64 }
-- output { true }
entry bdiv128Q4 [m] (uss0: [m][32][4]u64) (vss0: [m][32][4]u64) =
  #[unsafe]
  let uss = uss0 :> [m][32][2*2]u64
  let vss = vss0 :> [m][32][2*2]u64
  in  divTest uss vss
