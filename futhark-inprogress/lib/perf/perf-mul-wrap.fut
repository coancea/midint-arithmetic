import "../types"
import "../bmul"
import "../bmulWrap"

def bmulWGlb [m][q] (n: i64) (as: [m][2*q]u64) (bs: [m][2*q]u64) : [m][2*q]u64 =
  let areg = #[glb2reg_only(1)] manifest as
  let breg = #[glb2reg_only(1)] manifest bs
  in  bmulW n areg breg

--
-- ==
-- entry: oneFullMul2048
-- compiled random input { 2048i64 [32768][256][8]u64  [32768][256][8]u64 } 
entry oneFullMul2048 [m] (maxsz: i64) (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][256][2*4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][256][2*4]u64
   let bss = bss0 :> [m][256][2*4]u64
   let rss = imap2Intra ass bss (bmulWGlb maxsz)
   in  rss

--
-- ==
-- entry: oneHalfMul2048
-- compiled random input { 1024i64 [32768][256][8]u64  [32768][256][8]u64 } 
entry oneHalfMul2048 [m] (maxsz: i64) (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][256][2*4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][256][2*4]u64
   let bss = bss0 :> [m][256][2*4]u64
   let rss = imap2Intra ass bss (bmulWGlb maxsz)
   in  rss

--
-- ==
-- entry: oneFourthMul2048
-- compiled random input { 512i64 [32768][256][8]u64  [32768][256][8]u64 } 
entry oneFourthMul2048 [m] (maxsz: i64) (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][256][2*4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][256][2*4]u64
   let bss = bss0 :> [m][256][2*4]u64
   let rss = imap2Intra ass bss (bmulWGlb maxsz)
   in  rss

--
-- ==
-- entry: oneEightMul2048
-- compiled random input { 256i64 [32768][256][8]u64  [32768][256][8]u64 } 
entry oneEightMul2048 [m] (maxsz: i64) (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][256][2*4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][256][2*4]u64
   let bss = bss0 :> [m][256][2*4]u64
   let rss = imap2Intra ass bss (bmulWGlb maxsz)
   in  rss

