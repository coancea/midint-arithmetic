import "poly-uf"

-- ==
-- entry: poly256
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry poly256 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][1*64][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*64][2*2]u64
   let bss = bss0 :> [m][1*64][2*2]u64
   let rss = poly ass bss
   in  rss

