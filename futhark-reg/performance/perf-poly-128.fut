import "poly-uf"

-- ==
-- entry: poly128
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry poly128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][2*32][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][2*32][2*2]u64
   let bss = bss0 :> [m][2*32][2*2]u64
   let rss = poly ass bss
   in  rss
