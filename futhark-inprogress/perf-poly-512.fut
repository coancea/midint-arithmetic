import "poly"

-- ==
-- entry: poly512
-- compiled random input { [131072][128][4]u64  [131072][128][4]u64 }
entry poly512 [m] (ass0: [m][128][4]u64) (bss0: [m][128][4]u64) : [m][1*128][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*128][2*2]u64
   let bss = bss0 :> [m][1*128][2*2]u64
   let rss = poly ass bss
   in  rss

