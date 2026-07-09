import "../lib/types"
import "../lib/bdiv"

def shinvWrap [m][q] (vs: [m][2*q]uint) (h: i32) : [m][2*q]uint =
  let vreg = #[glb2reg_only(1)] manifest vs
  in  shinv vreg h

-- ==
-- entry: shinv2048Q8
-- compiled input @ data-div/data-shinv-1-2048-u64.in
-- output @ data-div/data-shinv-1-2048-u64.out
--
-- compiled input @ data-div/data-bin-shinv-1024-2048-u64.in
-- output @ data-div/data-bin-shinv-1024-2048-u64.out
--
entry shinv2048Q8 [n] (vs0: [n][2048]u64) (hs: [n]i32) : [n][2048]u64 =
  let vs = vs0 :> [n][256*(2*4)]u64
  let vs = opaque <| map unflatten vs
  let rs = opaque <| imap2Intra vs hs shinvWrap
  in  (map flatten rs) :> [n][2048]u64

