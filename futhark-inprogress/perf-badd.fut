import "badd"

-- ==
-- entry: oneAddition2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } auto output
entry oneAddition2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][2*256][4]u64 = #[unsafe]
   let ass = ass0 :> [m][(2*256)][4]u64
   let bss = bss0 :> [m][(2*256)][4]u64
   let rss = imap2Intra ass bss badd
   in  rss

-- ==
-- entry: sixAdditions1024
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 } auto output
entry sixAdditions1024 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : [m][2*256][4]u64 = #[unsafe]
   let ass = ass0 :> [m][(2*256)][4]u64
   let bss = bss0 :> [m][(2*256)][4]u64
   let apb   = imap2Intra ass bss badd     -- a + b
   let a2pb2 = imap2Intra apb apb badd     -- 2 * (a + b) = 2a + 2b
   let a2pb3 = imap2Intra a2pb2 bss badd   -- 2 * (a + b) + b = 2a + 3b
   let a4pb6 = imap2Intra a2pb3 a2pb3 badd -- 2 * (2a + 3b) = 4a + 6b
   let a6pb9 = imap2Intra a4pb6 a2pb3 badd -- (4a + 6b) + (2a + 3b) = 6a + 9b
   let rss   = imap2Intra a6pb9 bss badd   -- 6a + 10b
   in  rss


-- ==
-- entry: sixAdditions1024Q8
-- compiled random input { [32768][256][8]u64  [32768][256][8]u64 } auto output
entry sixAdditions1024Q8 [m] (ass0: [m][256][8]u64) (bss0: [m][256][8]u64) : [m][1*256][8]u64 = 
   #[unsafe]
   let ass = ass0 :> [m][(1*256)][8]u64
   let bss = bss0 :> [m][(1*256)][8]u64
   let apb   = imap2Intra ass bss badd     -- a + b
   let a2pb2 = imap2Intra apb apb badd     -- 2 * (a + b) = 2a + 2b
   let a2pb3 = imap2Intra a2pb2 bss badd   -- 2 * (a + b) + b = 2a + 3b
   let a4pb6 = imap2Intra a2pb3 a2pb3 badd -- 2 * (2a + 3b) = 4a + 6b
   let a6pb9 = imap2Intra a4pb6 a2pb3 badd -- (4a + 6b) + (2a + 3b) = 6a + 9b
   let rss   = imap2Intra a6pb9 bss badd   -- 6a + 10b
   in  rss
