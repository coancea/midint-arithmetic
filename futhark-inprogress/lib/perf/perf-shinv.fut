import "lib/github.com/diku-dk/cpprandom/random"
import "../types"
import "../shinv"

---- Entry point to generate 'n' random i32 elements
--entry main (seed: i32) (n: i64) : []i32 =
--  -- 1. Initialize the master RNG engine state
--  let rng_master = minstd_rand.init seed
--  
--  -- 2. Split the master state into 'n' independent parallel states
--  let rngs = minstd_rand.split_rng n rng_master
--  
--  -- 3. Map over the state array to generate a number for each index
--  let (_, random_numbers) = 
--    unzip (map (\rng_state -> minstd_rand.rand rng_state) rngs)
--    
--  in random_numbers
--

module dist = uniform_int_distribution i32 minstd_rand

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
--  let rng = dist.engine.rng_from_seed [1]
--  let rng_master = minstd_rand.init 1i32
--  let rng_state  = rng_master
  let rng = minstd_rand.rng_from_seed [1i32]
  let rngs= minstd_rand.split_rng num_instances rng
  let max_prec   = i32.i64 <| m*q2 - 2
  let (_, vss) = unzip <| map (mkRandBIntPart max_prec m q2) rngs
  in  (map unflatten vss, replicate num_instances max_prec)
--
--  let vss = #[scratch] replicate num_instances <| #[scratch]replicate (m*q2) zero_uint
--  let (_, vss) =
--    loop (rng, vss) for i < num_instances do
--      let (rng, row) = mkRandBIntPart max_prec m q2 rng
--      let vss[i] = row
--      in  (rng, vss)
--  in (map unflatten vss, replicate num_instances max_prec)
  
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

--
-- ==
-- entry: shinv4096Q4
-- "Shinv4096Q4" script input { mkShinvInput 16384i64 1024i64 4i64 }
entry shinv4096Q4 [m] (vss0: [m][1024][4]u64) (hs: [m]i32) : [m][1024][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][1024][2*2]u64
  in  imap2Intra vss hs shinvWrap

  
--
---- ==
-- entry: shinv2048Q8
-- "Shinv2048Q8" script input { mkShinvInput 32768 256 8 } 
--entry shinv2048Q8 [m] (vss0: [m][256][8]u64) (hs: [m]i32) : [m][256][2*4]u64 =
--  #[unsafe]
--  let vss = vss0 :> [m][256][2*4]u64
--  in  imap2Intra vss hs shinvWrap

--
-- ==
-- entry: shinv2048Q4
-- "Shinv2048Q4" script input { mkShinvInput 32768 512 4 } 
entry shinv2048Q4 [m] (vss0: [m][512][4]u64) (hs: [m]i32) : [m][512][2*2]u64 =
  #[unsafe]
  let vss = vss0 :> [m][512][2*2]u64
  in  imap2Intra vss hs shinvWrap

