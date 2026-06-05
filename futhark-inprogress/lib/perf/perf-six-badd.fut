import "../types"
import "../badd"

def sixAdd [n][ipb][m][q] (ass: [n][ipb*m][q]u64) (bss: [n][ipb*m][q]u64) : [n][ipb*m][q]u64 =
  #[unsafe]
  let apb   = imap2Intra ass bss badd     -- a + b
  let a2pb2 = imap2Intra apb apb badd     -- 2 * (a + b) = 2a + 2b
  let a2pb3 = imap2Intra a2pb2 bss badd   -- 2 * (a + b) + b = 2a + 3b
  let a4pb6 = imap2Intra a2pb3 a2pb3 badd -- 2 * (2a + 3b) = 4a + 6b
  let a6pb9 = imap2Intra a4pb6 a2pb3 badd -- (4a + 6b) + (2a + 3b) = 6a + 9b
  let rss   = imap2Intra a6pb9 bss badd   -- 6a + 10b
  in  rss

--
-- ==
-- entry: sixAdds4096
-- compiled random input { [16384][1024][4]u64  [16384][1024][4]u64 } 
entry sixAdds4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][4]u64 = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][4]u64
  let bss = bss0 :> [m][1*1024][4]u64
  in  sixAdd ass bss

-- ==
-- entry: sixAdds2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } auto output
entry sixAdds2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][1*512][4]u64 = #[unsafe]
   let ass = ass0 :> [m][1*512][4]u64
   let bss = bss0 :> [m][1*512][4]u64
   in  sixAdd ass bss

-- ==
-- entry: sixAdds1024
-- compiled random input { [65536][256][4]u64  [65536][256][4]u64 } 
entry sixAdds1024 [m] (ass0: [m][256][4]u64) (bss0: [m][256][4]u64) : [m][1*256][4]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][1*256][4]u64
   let bss = bss0 :> [m][1*256][4]u64
   in  sixAdd ass bss

--
-- ==
-- entry: sixAdds512
-- compiled random input { [131072][128][4]u64  [131072][128][4]u64 }
entry sixAdds512 [m] (ass0: [m][128][4]u64) (bss0: [m][128][4]u64) : [m][1*128][4]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*128][4]u64
   let bss = bss0 :> [m][1*128][4]u64
   in  sixAdd ass bss

--
-- ==
-- entry: sixAdds256
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry sixAdds256 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][1*64][4]u64 =
   #[unsafe]
   let ass = ass0 :> [m][1*64][4]u64
   let bss = bss0 :> [m][1*64][4]u64
   in  sixAdd ass bss

--
-- ==
-- entry: sixAdds128
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry sixAdds128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][2*32][4]u64 =
   #[unsafe]
   let ass = ass0 :> [m][2*32][4]u64
   let bss = bss0 :> [m][2*32][4]u64
   in  sixAdd ass bss

--
-- ==
-- entry: sixAdds64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 } auto output
entry sixAdds64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][4*16][4]u64 =
   #[unsafe]
   let ass = ass0 :> [m][4*16][4]u64
   let bss = bss0 :> [m][4*16][4]u64
   in  sixAdd ass bss

