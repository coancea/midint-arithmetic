import "badd"
import "classic-mul"

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
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry oneMultiply128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][2*32][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][2*32][2*2]u64
   let bss = bss0 :> [m][2*32][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

--
-- ==
-- entry: oneMultiply64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
entry oneMultiply64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : [m][4*16][2*2]u64 =
   #[unsafe]
   let ass = ass0 :> [m][4*16][2*2]u64
   let bss = bss0 :> [m][4*16][2*2]u64
   let rss = imap2Intra ass bss bmul
   in  rss

-----------------------------------------------
--- Polynomial: (a^2 + b) * (b^2 + b) + a*b
-----------------------------------------------

--
--def poly [m][ipb][n][q] (ass: [m][ipb*n][2*q]u64) (bss: [m][ipb*n][2*q]u64) : [m][ipb*n][2*q]u64 = 
--  #[unsafe]
--  imap2Intra ass bss
--    (\ a b ->
--        let a2   = bmul a  a            -- a^2
--        let a2pb = badd a2 b    -- a^2 + b
--        let b2   = bmul b  b            -- b^2
--        let b2pb = badd b2 b    -- b^2 + b
--        let prod = bmul a2pb b2pb       -- (a^2 + b) * (b^2 + b)
--        let ab   = bmul a  b            -- a*b
--        let res  = badd prod ab -- (a^2 + b) * (b^2 + b) + a*b
--        in  res
--    )
--

--
---- ==
-- entry: poly4096
-- compiled random input { [65536][1024][4]u64  [65536][1024][4]u64 } auto output
--entry poly4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : [m][1*1024][2*2]u64 = 
--  #[unsafe]
--  let ass = ass0 :> [m][1*1024][2*2]u64
--  let bss = bss0 :> [m][1*1024][2*2]u64
--  in  poly ass bss

