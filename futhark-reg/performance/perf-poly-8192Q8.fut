import "poly-uf"

-- ==
-- entry: poly8192Q8
-- compiled random input { [8192][1024][8]u64  [8192][1024][8]u64 } auto output
entry poly8192Q8 [m] (ass0: [m][1024][8]u64) (bss0: [m][1024][8]u64) : [m][1*1024][2*4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*4]u64
  let bss = bss0 :> [m][1*1024][2*4]u64
  in  poly ass bss

