import "poly"

-- ==
-- entry: poly1024
-- compiled random input { [65536][256][4]u64  [65536][256][4]u64 } 
entry poly1024 [m] (ass0: [m][256][4]u64) (bss0: [m][256][4]u64) : [m][1*256][2*2]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*256][2*2]u64
   let bss = bss0 :> [m][1*256][2*2]u64
   let rss = poly ass bss
   in  rss

