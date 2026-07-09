import "divSanityTest"

---------------------------------------------------------------------------
--- VERY STRANGE: With multiple entrypoints it runs out of shared memory
---                  AND DOES NOT VALIDATE !!!
---               With one entrypoint it does not run out of shared memory
---                  AND IT VALIDATES, i.e., u == v * quo + rem
---------------------------------------------------------------------------

entry mkShinvInput = mkShinvInput0

-- ==
-- entry: bdiv2048Q4
-- "Bdiv2048Q4" script input { mkShinvInput 32768i64 512i64 4i64 }
-- output { true }
entry bdiv2048Q4 [m] (uss0: [m][512][4]u64) (vss0: [m][512][4]u64) =
  #[unsafe]
  let uss = uss0 :> [m][512][2*2]u64
  let vss = vss0 :> [m][512][2*2]u64
  in  divTest uss vss
