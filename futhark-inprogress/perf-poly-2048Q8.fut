import "poly-uf"

-- ==
-- entry: poly2048Q8
-- compiled random input { [32768][256][8]u64  [32768][256][8]u64 } 
entry poly2048Q8 [m] (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][1*256][2*4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*256][2*4]u64
   let bss = bss0 :> [m][1*256][2*4]u64
   in  poly ass bss

