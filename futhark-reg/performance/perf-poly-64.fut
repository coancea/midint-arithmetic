import "poly-uf"

-- ==
-- entry: poly64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry poly64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][4*16][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][4*16][2*2]u64
   let bss = bss0 :> [m][4*16][2*2]u64
   let rss = poly ass bss
   in  rss

