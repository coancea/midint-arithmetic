import "../lib/types"
import "../lib/badd"
import "../lib/bmul"

def poly [m][ipb][n][q] (ass: [m][ipb*n][2*q]u64) (bss: [m][ipb*n][2*q]u64) : [m][ipb*n][2*q]u64 = 
  #[unsafe]
  let a2pbs = imap2Intra ass bss (\ a b -> let a2 = bmul a a in badd a2 b)
  let b2pbs = imap2Intra ass bss (\ a b -> let b2 = bmul b b in badd b2 a) -- should be badd b2 b
  let prods = imap2Intra a2pbs b2pbs (\ a2pb b2pb -> bmul a2pb b2pb)
  in  imap3Intra ass bss prods (\ a b prod -> let ab = bmul a b in badd prod ab)

-- ==
-- entry: poly8192Q8
-- compiled random input { [8192][1024][8]u64  [8192][1024][8]u64 } auto output
entry poly8192Q8 [m] (ass0: [m][1024][8]u64) (bss0: [m][1024][8]u64) : [m][1*1024][2*4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*4]u64
  let bss = bss0 :> [m][1*1024][2*4]u64
  in  poly ass bss

