import "poly"

-- ==
-- entry: poly4096Q8
-- compiled random input { [16384][512][8]u64  [16384][512][8]u64 } auto output
entry poly4096Q8 [m] (ass0: [m][512][8]u64) (bss0: [m][512][8]u64) : [m][1*512][2*4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*512][2*4]u64
  let bss = bss0 :> [m][1*512][2*4]u64
  in  poly ass bss

