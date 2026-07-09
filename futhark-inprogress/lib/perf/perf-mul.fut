import "../types"
import "../bmul"

--
-- ==
-- entry: oneMultiply8192Q8
-- compiled random input { [8192][1024][8]u64  [8192][1024][8]u64 }
entry oneMultiply8192Q8 [m] (ass0: [m][1024][8]u64) (bss0: [m][1024][8]u64) : [m][1*1024][2*4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*4]u64
  let bss = bss0 :> [m][1*1024][2*4]u64
  let rss = imap2Intra ass bss bmul
  in  rss


--
-- ==
-- entry: oneMultiply4096Q8
-- compiled random input { [16384][512][8]u64  [16384][512][8]u64 }
entry oneMultiply4096Q8 [m] (ass0: [m][512][8]u64) (bss0: [m][512][8]u64) : [m][1*512][2*4]u64 =
  #[unsafe]
  let ass = ass0 :> [m][1*512][2*4]u64
  let bss = bss0 :> [m][1*512][2*4]u64
  let rss = imap2Intra ass bss bmul
  in  rss

--
-- ==
-- entry: oneMultiply4096
-- compiled random input { [16384][1024][4]u64  [16384][1024][4]u64 } 

-- auto output
entry oneMultiply4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][2*2]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*2]u64
  let bss = bss0 :> [m][1*1024][2*2]u64
  let rss = imap2Intra ass bss bmul
  in  rss

--
-- ==
-- entry: oneMultiply2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } 
entry oneMultiply2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][1*512][2*2]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*512][2*2]u64
   let bss = bss0 :> [m][1*512][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply1024
-- compiled random input { [65536][256][4]u64  [65536][256][4]u64 } 
entry oneMultiply1024 [m] (ass0: [m][256][4]u64) (bss0: [m][256][4]u64) : [m][1*256][2*2]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*256][2*2]u64
   let bss = bss0 :> [m][1*256][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply512
-- compiled random input { [131072][128][4]u64  [131072][128][4]u64 }
entry oneMultiply512 [m] (ass0: [m][128][4]u64) (bss0: [m][128][4]u64) : [m][1*128][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*128][2*2]u64
   let bss = bss0 :> [m][1*128][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply256
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry oneMultiply256 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][1*64][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*64][2*2]u64
   let bss = bss0 :> [m][1*64][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply128
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry oneMultiply128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][2*32][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][2*32][2*2]u64
   let bss = bss0 :> [m][2*32][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry oneMultiply64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][4*16][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][4*16][2*2]u64
   let bss = bss0 :> [m][4*16][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

