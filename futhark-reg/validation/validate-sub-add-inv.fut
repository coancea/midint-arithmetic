import "../lib/types"
import "../lib/badd"
import "../lib/bsub"

---------------------------------------------------------------------------
--- VERY STRANGE: with multiple entrypoints it cannot do memory merging;
---               with one entrypoint it does and gets the performance
---               both ways validate, i.e., a == a + b - b
---------------------------------------------------------------------------


def addThenSub [n][ipb][m][q] (ass: [n][ipb*m][q]u64) (bss: [n][ipb*m][q]u64) : bool =
  #[unsafe]
  let apbss = imap2Intra ass bss badd
  let ass'  = imap2Intra apbss bss bsub
  let ass'' = opaque ass'
  --
  in  map2 (==) (flatten (flatten ass)) (flatten (flatten ass''))
      |> reduce (&&) true

--
---- ==
-- entry: safe4096
-- compiled random input { [16384][1024][4]u64  [16384][1024][4]u64 }
-- output { true }
entry safe4096 [m] (ass0: [m][1024][4]u64) (bss0: [m][1024][4]u64) : bool = 
  #[unsafe]
  let ass = ass0 :> [m][1*1024][4]u64
  let bss = bss0 :> [m][1*1024][4]u64
  in  addThenSub ass bss

-- ==
-- entry: safe2048
-- compiled random input { [32768][512][4]u64  [32768][512][4]u64 }
-- output { true }
entry safe2048 [m] (ass0: [m][512][4]u64) (bss0: [m][512][4]u64) : bool =
   #[unsafe]
   let ass = ass0 :> [m][1*512][4]u64
   let bss = bss0 :> [m][1*512][4]u64
   in  addThenSub ass bss

-- ==
-- entry: safe1024
-- compiled random input { [65536][256][4]u64  [65536][256][4]u64 }
-- output { true }
entry safe1024 [m] (ass0: [m][256][4]u64) (bss0: [m][256][4]u64) : bool = 
   #[unsafe]
   let ass = ass0 :> [m][1*256][4]u64
   let bss = bss0 :> [m][1*256][4]u64
   in  addThenSub ass bss

--
-- ==
-- entry: safe512
-- compiled random input { [131072][128][4]u64  [131072][128][4]u64 }
-- output { true }
entry safe512 [m] (ass0: [m][128][4]u64) (bss0: [m][128][4]u64) : bool =
   #[unsafe]
   let ass = ass0 :> [m][1*128][4]u64
   let bss = bss0 :> [m][1*128][4]u64
   in  addThenSub ass bss

--
-- ==
-- entry: safe256
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
-- output { true }
entry safe256 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : bool =
   #[unsafe]
   let ass = ass0 :> [m][1*64][4]u64
   let bss = bss0 :> [m][1*64][4]u64
   in  addThenSub ass bss

--
-- ==
-- entry: safe128
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
-- output { true }
entry safe128 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : bool =
   #[unsafe]
   let ass = ass0 :> [m][2*32][4]u64
   let bss = bss0 :> [m][2*32][4]u64
   in  addThenSub ass bss

--
-- ==
-- entry: safe64
-- compiled random input { [262144][64][4]u64  [262144][64][4]u64 }
-- output { true }
entry safe64 [m] (ass0: [m][64][4]u64) (bss0: [m][64][4]u64) : bool =
   #[unsafe]
   let ass = ass0 :> [m][4*16][4]u64
   let bss = bss0 :> [m][4*16][4]u64
   in  addThenSub ass bss

