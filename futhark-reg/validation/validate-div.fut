import "../lib/types"
import "../lib/badd"
import "../lib/bmul"
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

def divTest [n][m][q] (uss: [n][m][2*q]u64) (vss: [n][m][2*q]u64) : bool =
   let (quo, rem) = unzip <| opaque <| imap2Intra uss vss bdiv
   let tmp = opaque <| imap2Intra (vss :> [n][1*m][2*q]u64) (quo :> [n][1*m][2*q]u64) bmul
   let uss'= opaque <| imap2Intra tmp (rem  :> [n][1*m][2*q]u64) badd
   let uss_flat = flatten (map flatten (uss :> [n][1*m][2*q]u64))
   let uss_flat'= flatten (map flatten uss')
   in  uss_flat == uss_flat' 

-- ==
-- entry: sanityBdiv2048Q4
-- compiled input @ data-div/data-div-1-2048-u64.in
-- output { true } 
--
-- compiled input @ data-div/data-bin-div-1024-2048-u64.in
-- output { true }
--
entry sanityBdiv2048Q4 [m] (uss0: [m][512][4]u64) (vss0: [m][512][4]u64) : bool =
  #[unsafe]
  let uss = uss0 :> [m][512][2*2]u64
  let vss = vss0 :> [m][512][2*2]u64
  in  divTest uss vss

