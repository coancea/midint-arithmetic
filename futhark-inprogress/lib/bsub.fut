import "types"
import "badd"

------------------------------------------------
---- Implements subtraction of big integers;
----   very similar to how addition is done!
------------------------------------------------

-- | Specialized case for subtraction: `B^h - as`
--   Assumes B^h >= as
def bsubFromBpowReg [m][q] (h: i32) (xs: [m][q]uint) : [m][q]uint =
  #[unsafe]
  let shm = replicate 1 i32.highest
  let ffacc (myacc: *acc ([1]i32)) (tid: i64) : acc ([1]i32) =
      let ind =
        loop ind = i32.highest for i < q do
          let rev_i = q - i - 1
          in  if xs[tid, rev_i] != 0
              then i32.i64 (q * tid + rev_i) else ind
      in write myacc 0 ind
  --
  let shm = opaque <|
    reduce_by_index_stream shm (i32.min) i32.highest ffacc (iota m)
  --
  let min_index = shm[0]
  --
  let ffin tid x i =
      let idx = i32.i64 (q * tid + i) in
      if idx < min_index then zero_uint
      else if idx < h
           then (highest_uint - x) + uint_bool (idx == min_index)
           else x
  let ffout xs tid =
      #[sequential]map2 (ffin tid) xs (iota q)
   in  opaque <| #[toregmem(1)] map2 ffout xss (iota m)


let bsubReg [ipb][n][q] (aregs : [ipb*n][q]uint) (bregs : [ipb*n][q]uint) : [ipb*n][q]uint =
  #[unsafe]
  let ff1 tid =
    let (areg, breg) = (aregs[tid], bregs[tid])
    let carry_acc = carryOpNE
    let rs = #[scratch]replicate q zero_uint
    let cs = #[scratch]replicate q carryOpNE
    let is_seg_start = (tid * q) % (q * n) == 0
    in  loop (carry_acc, rs, cs) for i < q do
          let (a, b) = (areg[i], breg[i])
          let r = a - b
          let c = cTfromBool (r > a)
          let c = c | ( (cTfromBool (r == zero_uint)) << 1 )
          let c = c | ( (cTfromBool (i == 0 && is_seg_start)) << 2 )
          let rs[i] = r
          let cs[i] = c         
          let carry_acc = carrySegOp carry_acc c
          in  (carry_acc, rs, cs)
  --
  let (carry_thds, rss, css) =
      opaque <| unzip3 <| #[toregmem(1)] map ff1 (iota (ipb*n))
  --
  let carry_thds = opaque <| scan carrySegOp carryOpNE carry_thds
  --
  let ff2 rs cs tid =
    #[unsafe]
    let is_seg_start = (tid * q) % (q * n) == 0
    let carry = if is_seg_start then carryOpNE else carry_thds[tid-1]
    let rs' = #[scratch] replicate q zero_uint
    let (rs', _) =
        loop (rs', carry) for i < q do
          let vb = cs[i] & 4 == 0 && carry & 1 == 1
          let rs'[i] = rs[i] - uint_bool vb
          let carry = carrySegOp carry cs[i]
          in  (rs', carry)
    in rs'
  --
  in  opaque <| #[toregmem(1)] map3 ff2 rss css (iota (ipb*n))

let bsub [ipb][n][q] (as : [ipb*n][q]uint) (bs : [ipb*n][q]uint) : [ipb*n][q]uint =
  let areg = #[glb2reg_only(1)] manifest as
  let breg = #[glb2reg_only(1)] manifest bs
  in  opaque <| bsubReg areg breg

