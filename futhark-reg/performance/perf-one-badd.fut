import "../lib/types"
import "../lib/badd"

--
-- ==
-- entry: oneAdd4096Q8
-- compiled random input { [16384][512][8]u64  [16384][512][8]u64 }
entry oneAdd4096Q8 [m] (ass0: [m][512][8]u64) (bss0: [m][512][8]u64) : [m][1*512][8]u64 =
  #[unsafe]
  let ass = ass0 :> [m][1*512][8]u64
  let bss = bss0 :> [m][1*512][8]u64
  in  imap2Intra ass bss badd

--
-- ==
-- entry: oneAdd4096
-- compiled random input { [16384][1024][4]u64  [16384][1024][4]u64 } 
entry oneAdd4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][4]u64
  let bss = bss0 :> [m][1*1024][4]u64
  in  imap2Intra ass bss badd

-- ==
-- entry: oneAdd2048Q8
-- compiled random input { [32768][256][8]u64  [32768][256][8]u64 } auto output
entry oneAdd2048Q8 [m] (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][1*256][8]u64 = #[unsafe]
  let ass = ass0 :> [m][(1*256)][8]u64
  let bss = bss0 :> [m][(1*256)][8]u64
  in  imap2Intra ass bss badd
   
-- ==
-- entry: oneAdd2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } auto output
entry oneAdd2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][1*512][4]u64 = #[unsafe]
  let ass = ass0 :> [m][1*512][4]u64
  let bss = bss0 :> [m][1*512][4]u64
  in  imap2Intra ass bss badd

-- ==
-- entry: oneAdd1024
-- compiled random input { [65536][256][4]u64  [65536][256][4]u64 } 
entry oneAdd1024 [m] (ass0: [m][256][4]u64) (bss0: [m][256][4]u64) : [m][1*256][4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*256][4]u64
  let bss = bss0 :> [m][1*256][4]u64
  in  imap2Intra ass bss badd

--
-- ==
-- entry: oneAdd512
-- compiled random input { [131072][128][4]u64  [131072][128][4]u64 }
entry oneAdd512 [m] (ass0: [m][128][4]u64) (bss0: [m][128][4]u64) : [m][1*128][4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][1*128][4]u64
  let bss = bss0 :> [m][1*128][4]u64
  in  imap2Intra ass bss badd

--
-- ==
-- entry: oneAdd256
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry oneAdd256 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][1*64][4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][1*64][4]u64
  let bss = bss0 :> [m][1*64][4]u64
  in  imap2Intra ass bss badd

--
-- ==
-- entry: oneAdd128
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry oneAdd128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][2*32][4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][2*32][4]u64
  let bss = bss0 :> [m][2*32][4]u64
  in  imap2Intra ass bss badd

--
-- ==
-- entry: oneAdd64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry oneAdd64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][4*16][4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][4*16][4]u64
  let bss = bss0 :> [m][4*16][4]u64
  in  imap2Intra ass bss badd

