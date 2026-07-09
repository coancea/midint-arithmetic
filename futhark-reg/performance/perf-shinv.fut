import "lib/github.com/diku-dk/cpprandom/random"
import "../lib/types"
import "../lib/bdiv"

def mkRandBIntFull (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64
  in
  loop (rng_state, arr) for i < max_prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

def mkRandBIntPart (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64
  let (rng_state, r) = minstd_rand.rand rng_state
  let prec = ( (i32.u32 r) % (max_prec/2) ) + 2
  in
  loop (rng_state, arr) for i < prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

--
---- ==
-- entry: mkShinvInput
-- compiled input { 32768i64 256i64 8i64 }
entry mkShinvInput (num_instances: i64) (m: i64) (q2: i64)
                 : ( [num_instances][m][q2]u64
                   , [num_instances]i32
                   ) =
  let rng = minstd_rand.rng_from_seed [1i32]
  let rngs= minstd_rand.split_rng num_instances rng
  let max_prec   = i32.i64 <| m*q2 - 2
  let (_, vss) = unzip <| map (mkRandBIntPart max_prec m q2) rngs
  in  (map unflatten vss, replicate num_instances max_prec)
  
-------------------------------------------------------------
--- entry points for evaluating the performance of shinv
-------------------------------------------------------------

def shinvWrap [m][q] (vs: [m][2*q]uint) (h: i32) : [m][2*q]uint =
  let vreg = #[glb2reg_only(1)] manifest vs
  in  shinv vreg h

--
---- ==
-- entry: shinv4096Q8
-- "Shinv4096Q8" script input { mkShinvInput 16384 512 8 }
--entry shinv4096Q8 [m] (vss0: [m][512][8]u64) (hs: [m]i32) : [m][512][2*4]u64 =
--  #[unsafe]
--  let vss = vss0 :> [m][512][2*4]u64
--  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv4096Q4
-- "Shinv4096Q4" script input { mkShinvInput 16384i64 1024i64 4i64 }
entry shinv4096Q4 [m] (vss0: [m][1024][4]u64) (hs: [m]i32) : [m][1024][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][1024][2*2]u64
  in  imap2Intra vss hs shinvWrap

---- ==
-- entry: shinv2048Q8
-- "Shinv2048Q8" script input { mkShinvInput 32768 256 8 } 
--entry shinv2048Q8 [m] (vss0: [m][256][8]u64) (hs: [m]i32) : [m][256][2*4]u64 =
--  #[unsafe]
--  let vss = vss0 :> [m][256][2*4]u64
--  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv2048Q4
-- "Shinv2048Q4" script input { mkShinvInput 32768 512 4 } 
entry shinv2048Q4 [m] (vss0: [m][512][4]u64) (hs: [m]i32) : [m][512][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][512][2*2]u64
  in  imap2Intra vss hs shinvWrap

---- ==
-- entry: shinv1024Q4
-- "Shinv1024Q4" script input { mkShinvInput 65536 256 4 } 
--entry shinv1024Q4 [m] (vss0: [m][256][4]u64) (hs: [m]i32) : [m][256][2*2]u64 =
--  #[unsafe]
--  let vss = vss0 :> [m][256][2*2]u64
--  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv1024Q8
-- "Shinv1024Q8" script input { mkShinvInput 65536 128 8 } 
entry shinv1024Q8 [m] (vss0: [m][128][8]u64) (hs: [m]i32) : [m][128][2*4]u64 =
  #[unsafe]
  let vss = vss0 :> [m][128][2*4]u64
  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv512Q4
-- "Shinv512Q4" script input { mkShinvInput 131072 128 4 } 
entry shinv512Q4 [m] (vss0: [m][128][4]u64) (hs: [m]i32) : [m][128][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][128][2*2]u64
  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv256Q4
-- "Shinv256Q4" script input { mkShinvInput 262144 64 4 } 
entry shinv256Q4 [m] (vss0: [m][64][4]u64) (hs: [m]i32) : [m][64][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][64][2*2]u64
  in  imap2Intra vss hs shinvWrap

-- ==
-- entry: shinv128Q4
-- "Shinv128Q4" script input { mkShinvInput 524288 32 4 } 
entry shinv128Q4 [m] (vss0: [m][32][4]u64) (hs: [m]i32) : [m][32][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][32][2*2]u64
  in  imap2Intra vss hs shinvWrap
