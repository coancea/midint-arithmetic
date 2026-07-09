import "../lib/types"
import "../lib/bdiv"

-- ==
-- entry: bdiv2048Q4
-- compiled input @ data-div/data-div-1-2048-u64.in
-- output @ data-div/data-div-1-2048-u64.out
--
-- compiled input @ data-div/data-bin-div-1024-2048-u64.in
-- output @ data-div/data-bin-div-1024-2048-u64.out
--

entry bdiv2048Q4 [m] (uss0: [m][512][4]u64) (vss0: [m][512][4]u64) : ([m][512][2*2]u64, [m][512][2*2]u64) =
  #[unsafe]
  let uss = uss0 :> [m][512][2*2]u64
  let vss = vss0 :> [m][512][2*2]u64
  in  unzip <| imap2Intra uss vss bdiv
