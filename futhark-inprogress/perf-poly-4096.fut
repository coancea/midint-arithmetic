import "poly-uf"

-- ==
-- entry: poly4096
-- compiled random input { [16384][1024][4]u64  [16384][1024][4]u64 } auto output
entry poly4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][2*2]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*2]u64
  let bss = bss0 :> [m][1*1024][2*2]u64
  in  poly ass bss

