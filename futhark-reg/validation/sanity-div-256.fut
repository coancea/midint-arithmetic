import "divSanityTest"

entry mkShinvInput = mkShinvInput0

-- ==
-- entry: bdiv256Q4
-- "Bdiv256Q4" script input { mkShinvInput 262144i64 64i64 4i64 }
-- output { true }
entry bdiv256Q4 [m] (uss0: [m][64][4]u64) (vss0: [m][64][4]u64)  =
  #[unsafe]
  let uss = uss0 :> [m][64][2*2]u64
  let vss = vss0 :> [m][64][2*2]u64
  in  divTest uss vss
