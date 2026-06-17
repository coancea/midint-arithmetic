import "mkDivData"
import "../shinv"

entry mkShinvInput = mkShinvInput'

-- ==
-- entry: bdiv4096Q4
-- "Bdiv4096Q4" script input { mkShinvInput 16384i64 1024i64 4i64 }
entry bdiv4096Q4 [m] (uss: [m][1024][4]u64) (vss: [m][1024][4]u64) =
  --  : ([m][1024][2*2]u64, [m][1024][2*2]u64) =
  --#[unsafe] 
  bdivBatch (uss :> [m][1024][2*2]u64) (vss :> [m][1024][2*2]u64)
