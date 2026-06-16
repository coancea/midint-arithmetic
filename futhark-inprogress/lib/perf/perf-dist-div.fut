import "lib/github.com/diku-dk/cpprandom/random"
import "../types"
import "../shinv"

def mkRandBIntFull (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64 in
  loop (rng_state, arr) for i < max_prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

def mkRandBIntPart (max_prec: i32) (m: i64) (q2: i64) rng_state =
  let arr = replicate (m*q2) 0u64
  let (rng_state, r) = minstd_rand.rand rng_state
  let prec = ( (i32.u32 r) % (max_prec/2) ) + 2 in
  loop (rng_state, arr) for i < prec do
    let (rng_state, v) = minstd_rand.rand rng_state
    let arr[i] = u64.u32 v
    in  (rng_state, arr)

---- ==
-- entry: mkShinvInput
-- compiled input { 32768i64 256i64 8i64 }
entry mkShinvInput (num_instances: i64) (m: i64) (q2: i64)
                 : ( [num_instances][m][q2]u64
                   , [num_instances][m][q2]u64
                   ) =
  let rng = minstd_rand.rng_from_seed [1i32]
  let rngs= minstd_rand.split_rng num_instances rng
  let max_prec   = i32.i64 <| m*q2 - 2
  let (rngs', uss) = unzip <| map (mkRandBIntFull max_prec m q2) rngs
  let (_,     vss) = unzip <| map (mkRandBIntPart max_prec m q2) rngs'
  in  (map unflatten uss, map unflatten vss)  -- replicate num_instances max_prec
  
-------------------------------------------------------------
--- entry points for evaluating the performance of shinv
-------------------------------------------------------------

-- ==
-- entry: bdiv4096Q4
-- "Bdiv4096Q4" script input { mkShinvInput 16384i64 1024i64 4i64 }
entry bdiv4096Q4 [m] (uss: [m][1024][4]u64) (vss: [m][1024][4]u64) : ([m][1024][2*2]u64, [m][1024][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][1024][2*2]u64) (vss :> [m][1024][2*2]u64)

-- ==
-- entry: bdiv2048Q4
-- "Bdiv2048Q4" script input { mkShinvInput 32768i64 512i64 4i64 }
entry bdiv2048Q4 [m] (uss: [m][512][4]u64) (vss: [m][512][4]u64) : ([m][512][2*2]u64, [m][512][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][512][2*2]u64) (vss :> [m][512][2*2]u64)

-- ==
-- entry: bdiv1024Q4
-- "Bdiv1024Q4" script input { mkShinvInput 65536i64 256i64 4i64 }
entry bdiv1024Q4 [m] (uss: [m][256][4]u64) (vss: [m][256][4]u64) : ([m][256][2*2]u64, [m][256][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][256][2*2]u64) (vss :> [m][256][2*2]u64)

-- ==
-- entry: bdiv1024Q8
-- "Bdiv1024Q8" script input { mkShinvInput 65536i64 128i64 8i64 }
entry bdiv1024Q8 [m] (uss: [m][128][8]u64) (vss: [m][128][8]u64) : ([m][128][2*4]u64, [m][128][2*4]u64) =
  #[unsafe] bdivBatch (uss :> [m][128][2*4]u64) (vss :> [m][128][2*4]u64)

-- ==
-- entry: bdiv512Q4
-- "Bdiv512Q4" script input { mkShinvInput 131072i64 128i64 4i64 }
entry bdiv512Q4 [m] (uss: [m][128][4]u64) (vss: [m][128][4]u64) : ([m][128][2*2]u64, [m][128][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][128][2*2]u64) (vss :> [m][128][2*2]u64)

-- ==
-- entry: bdiv256Q4
-- "Bdiv256Q4" script input { mkShinvInput 262144i64 64i64 4i64 }
entry bdiv256Q4 [m] (uss: [m][64][4]u64) (vss: [m][64][4]u64) : ([m][64][2*2]u64, [m][64][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][64][2*2]u64) (vss :> [m][64][2*2]u64)

-- ==
-- entry: bdiv128Q4
-- "Bdiv128Q4" script input { mkShinvInput 524288i64 32i64 4i64 }
entry bdiv128Q4 [m] (uss: [m][32][4]u64) (vss: [m][32][4]u64) : ([m][32][2*2]u64, [m][32][2*2]u64) =
  #[unsafe] bdivBatch (uss :> [m][32][2*2]u64) (vss :> [m][32][2*2]u64)
