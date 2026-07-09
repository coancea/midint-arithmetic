import "poly-uf"

-- ==
-- entry: poly2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } 
entry poly2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][1*512][2*2]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*512][2*2]u64
   let bss = bss0 :> [m][1*512][2*2]u64
   in  poly ass bss

