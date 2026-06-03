import "badd"

def sixAdd [n][ipb][m][q] (ass: [n][ipb*m][q]u64) (bss: [n][ipb*m][q]u64) : [n][ipb*m][q]u64 =
  #[unsafe]
  let apb   = imap2Intra ass bss badd     -- a + b
  let a2pb2 = imap2Intra apb apb badd     -- 2 * (a + b) = 2a + 2b
  let a2pb3 = imap2Intra a2pb2 bss badd   -- 2 * (a + b) + b = 2a + 3b
  let a4pb6 = imap2Intra a2pb3 a2pb3 badd -- 2 * (2a + 3b) = 4a + 6b
  let a6pb9 = imap2Intra a4pb6 a2pb3 badd -- (4a + 6b) + (2a + 3b) = 6a + 9b
  let rss   = imap2Intra a6pb9 bss badd   -- 6a + 10b
  in  rss

-- ==
-- entry: sixAdds2048
-- compiled random input { [32768][256][8]u64  [32768][256][8]u64 } auto output
entry sixAdds2048 [m] (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][1*256][8]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][(1*256)][8]u64
   let bss = bss0 :> [m][(1*256)][8]u64
   in  sixAdd ass bss
