import "types"

------------------------------------------------------------------------
---- prefix sum (scan) operator to propagate the carry
-- let add_op (ov1 : bool, mx1: bool) (ov2 : bool, mx2: bool) : (bool, bool) =
--   ( (ov1 && mx2) || ov2,    mx1 && mx2 )
------------------------------------------------------------------------

---- prefix sum (scan) operator to propagate the curry:
---- format: last digit set      => overfolow
----         ante-last digit set => one unit away from overflowing   
let carryOp (c1: cT) (c2: cT) =
  (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1)
  
let carrySegOp (c1: cT) (c2: cT) =
    if (c2 & 4) != 0 then c2
    else let res = ( (c1 & (c2 >> 1)) | c2 ) & 1
         let res = res | (c1 & c2  & 2)
         in  ( res | ( (c1 | c2) & 4 ) )

let baddReg [ipb][n][q] (aregs : [ipb*n][q]uint) (bregs : [ipb*n][q]uint) : [ipb*n][q]uint =
  #[unsafe]
  let ff1 tid =
    let (areg, breg) = (aregs[tid], bregs[tid])
    let carry_acc = carryOpNE
    let rs = #[scratch]replicate q zero_uint
    let cs = #[scratch]replicate q carryOpNE
    let is_seg_start = (tid * q) % (q * n) == 0
    in  loop (carry_acc, rs, cs) for i < q do
          let (a, b) = (areg[i], breg[i])
          let r = a + b
          let c = cTfromBool (r < a)
          let c = c | ( (cTfromBool (r == uint_highest)) << 1 )
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
          let rs'[i] = rs[i] + uint_bool vb
          let carry = carrySegOp carry cs[i]
          in  (rs', carry)
    in rs'
  --
  in  opaque <| #[toregmem(1)] map3 ff2 rss css (iota (ipb*n))


let badd [ipb][n][q] (as : [ipb*n][q]uint) (bs : [ipb*n][q]uint) : [ipb*n][q]uint =
  let areg = #[glb2reg_only(1)] manifest as
  let breg = #[glb2reg_only(1)] manifest bs
  in  opaque <| baddReg areg breg

