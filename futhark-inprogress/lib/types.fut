
-----------------------------------------------------------
--- Types for the type of scan operator in big addition ---
-----------------------------------------------------------

type cT         = u32      --u8
let  cTfromBool = u32.bool --u8.bool
let  carryOpNE: cT = 2u32   --2u8

--type cT         = u8
--let  cTfromBool = u8.bool
--let  two_cT     = 2u8

------------------------------------
--- Types for big multiplication ---
------------------------------------

type uint  = u64
let uint_mul_hi = u64.mul_hi
let uint_bool = u64.bool
let zero_uint = 0u64
let one_uint  = 1u64
let uint_highest = u64.highest
let uint_cT   = u64.u32

--type D = u64
--let D_mul_hi = u64.mul_hi
--let D_bool = u64.bool
--let zeroD = 0u64

type uint_c = u32
let zero_c = 0u32
let size_c = 32u64
let c_bool = u32.bool
let c_uint = u32.u64
let uint_c = u64.u32

--type S = u32
--let zeroS = 0u32
--let lenS = 32u64
--let S_bool = u32.bool
--let S_D = u32.u64
--let D_S = u64.u32
--type Dx4   = (D,D,D,D)

---------------------------------
--- SOAC renaming
---------------------------------

let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap3 as bs cs f = map3 f as bs cs

let imapIntra as f = #[incremental_flattening(only_intra)] map f as
let imap2Intra as bs f = #[incremental_flattening(only_intra)] map2 f as bs
let imap3Intra as bs cs f = #[incremental_flattening(only_intra)] map3 f as bs cs

