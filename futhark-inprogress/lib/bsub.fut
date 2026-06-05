import "types"
import "badd"

------------------------------------------------
---- Implements subtraction of big integers;
----   very similar to how addition is done!
------------------------------------------------

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

