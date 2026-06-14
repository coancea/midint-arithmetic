import "../types"
import "../bmulFull"
import "../bmul"

--
-- ==
-- entry: oneDiv4096
-- compiled random input { 512i64 [16384][1024][4]u64  [16384][1024][4]u64 } 
entry oneDiv4096 [m] (h: i64) (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][2*2]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][2*2]u64
  let bss = bss0 :> [m][1*1024][2*2]u64
  let rss = imap2Intra ass bss (bmulSftFullRegs h)
  in  rss

--
-- ==
-- entry: oneDiv4096Q8
-- compiled random input { 512i64 [16384][512][8]u64  [16384][512][8]u64 } 
entry oneDiv4096Q8 [m] (h: i64) (ass0: [m][512][8]u64) (bss0: [m][512][8]u64) : [m][1*512][2*4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*512][2*4]u64
  let bss = bss0 :> [m][1*512][2*4]u64
  let rss = imap2Intra ass bss (bmulSftFullRegs h)
  in  rss

--
-- ==
-- entry: oneDiv2048
-- compiled random input { 256i64 [32768][512][4]u64  [32768][512][4]u64 } 
entry oneDiv2048 [m] (h: i64) (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][1*512][2*2]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*512][2*2]u64
   let bss = bss0 :> [m][1*512][2*2]u64
   let rss = imap2Intra ass bss (bmulSftFullRegs h)
   in  rss
