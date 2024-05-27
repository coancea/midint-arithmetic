def ceilLg (n: i64) : i32 =
  let (last_seen, num_seen, _, _) =
    loop (last_seen, num_seen, i,    m) = 
         (0i32,      0i32,     0i32, n)
    while m > 0 do
      let (last_seen', num_seen') =
          if ( (i32.i64 m) & 1 ) == 1
          then (i, num_seen + 1)
          else (last_seen, num_seen)
      in  (last_seen', num_seen', i+1, m >> 1)
  in  last_seen + i32.bool (num_seen > 1)

def bitReverse (j: i32) (nb: i32) =
  let r = 0i32
  let (r,_) =
    loop (r,j) for _k < nb do
      let r = (r << 1) | (j & 1)
      in  (r, j >> 1)
  in  r

module type UZ = {
   module mus: integral
   module mss: integral
   module mud: integral
   
   type ustp

   val m: mus.t
   val n: i32
   val k: mss.t
   val g: mus.t
   val b: mss.t
   val us0 : mus.t
   val us1 : mus.t
   
   val per : ustp  -> mus.t
   val rep : mus.t -> ustp
   val d_s : mus.t -> mud.t
   val s_d : mud.t -> mus.t
   val s_u : mus.t -> mss.t
   val u_s : mss.t -> mus.t
   val us_i32: i32 -> mus.t
}

module uz32 : UZ with ustp = u32 = {
  module mus = u32
  module mss = i32
  module mud = u64
  
  type ustp = u32
  
  def per (x: ustp) = x
  def rep (x: mus.t) = x
  def d_s = u64.u32
  def s_d = u32.u64
  def s_u = i32.u32
  def u_s = u32.i32
  def us_i32 = u32.i32
  
  def m = 3221225473u32
  def n = 30i32
  def k = 3i32
  def g = 13u32
  def b = 1i32 << 15
  def us0 = 0u32
  def us1 = 1u32
}

module type PrimeField = {
  module mus : integral  -- unsigned "single" integer
  module mss : integral  -- signed   "single" integer
  module mud : integral  -- unsigned "double" integer

  type ustp

  val m: mus.t
  val n: i32
  val k: mss.t
  val g: mus.t
  val b: mss.t
  val us0 : mus.t
  val us1 : mus.t

  val us_i32: i32 -> mus.t
  val per : ustp  -> mus.t
  val rep : mus.t -> ustp

  val +: mus.t -> mus.t -> mus.t
  val -: mus.t -> mus.t -> mus.t
  val *: mus.t -> mus.t -> mus.t
  val /: mus.t -> mus.t -> mus.t
  val ^: mus.t -> mus.t -> mus.t
  val inv: mus.t -> mus.t
  val neg: mus.t -> mus.t
}


module mk_zmod (uZ : UZ) : PrimeField with ustp = uZ.ustp = {
  module mus = uZ.mus
  module mss = uZ.mss
  module mud = uZ.mud

  type ustp = uZ.ustp
  
  def m = uZ.m
  def n = uZ.n
  def k = uZ.k
  def g = uZ.g
  def b = uZ.b
  def us0 = uZ.us0
  def us1 = uZ.us1
  
  def per = uZ.per
  def rep = uZ.rep
  def us_i32 = uZ.us_i32
  
  def (x: mus.t) * (y: mus.t) : mus.t =
    let r = (uZ.d_s x) uZ.mud.* (uZ.d_s y)
    in  r uZ.mud.% (uZ.d_s uZ.m) |> uZ.s_d

  def (x: mus.t) ^ (n: mus.t) : mus.t =
    if n uZ.mus.== us0 then us1 
    else 
      let (a,b,_) =
        loop (a, b, m) = (x, us1, n)
        while (m uZ.mus.> us1) do
          let b' = if (m uZ.mus.& us1) uZ.mus.== us1 then a * b else b
          let a' = a * a
          in (a', b', m uZ.mus.>> us1)
      in  a * b

  def (x: mus.t) - (y: mus.t) : mus.t =
    let r = if x uZ.mus.< y then x uZ.mus.+ uZ.m else x
    in  r uZ.mus.- y

  def (x: mus.t) + (y: mus.t) : mus.t =
    let r = (uZ.d_s x) uZ.mud.+ (uZ.d_s y)
    let md = uZ.d_s uZ.m 
    let r = if r uZ.mud.>= md
            then r uZ.mud.- md
            else r
    in  uZ.s_d r
    
  def inv (x: mus.t) : mus.t =
    let (a, b) = (x, uZ.m)
    let szero = uZ.s_u us0
    let (s, t) = (uZ.s_u us1, szero)
    let (s,_,_,_) = 
      loop (s,t,a,b) while (b uZ.mus.!= uZ.us0) do
          let (q,r) = (a uZ.mus./ b, a uZ.mus.% b)
          let a = b
          let b = r
          let r = s uZ.mss.- ( (uZ.s_u q) uZ.mss.* t )
          let s = t
          let t = r
          in  (s,t,a,b)
    let s = if (s uZ.mss.< szero) then s uZ.mss.+ (uZ.s_u uZ.m) else s
    in  uZ.u_s s

  def neg (x: mus.t) : mus.t =
    if x uZ.mus.!= uZ.us0
    then uZ.m uZ.mus.- x
    else uZ.us0

  def (x: mus.t) / (y: mus.t) : mus.t = x * (inv y)
}


module type fftz1d = {
  -- the prime field
  module pf : PrimeField
  
  type usereltp
  
  val elper : usereltp -> pf.mus.t
  val elrep : pf.mus.t -> usereltp
  
  val getOmega : i64 -> pf.mus.t
    
  val permute_seq [n]: *[n]pf.mus.t -> *[n]pf.mus.t
  val permute_par [n]:  [n]pf.mus.t -> *[n]pf.mus.t

  val fft2_seq [n]: pf.mus.t -> [n]pf.mus.t -> [n]pf.mus.t
  val fft2_par [n]: pf.mus.t -> [n]pf.mus.t -> [n]pf.mus.t
  
  val ffft2 [n]: [n]pf.mus.t -> [n]pf.mus.t
  val ifft2 [n]: [n]pf.mus.t -> [n]pf.mus.t
  
  val fftmul[n]: [n]pf.mus.t -> [n]pf.mus.t -> [n]pf.mus.t 
}

module mk_fftz1d (pfa: PrimeField) : fftz1d with usereltp = pfa.ustp = {
  module pf = pfa
  
  type usereltp = pfa.ustp
  
  def elper = pf.per
  def elrep = pf.rep

  def getOmega (n: i64) = 
    let shft: pf.mus.t = pf.us_i32 (pf.n - (ceilLg n))
    let e = (pf.us1) pf.mus.<< shft
    in  pf.g pf.^ e

  def is_power_of_2 (x: i64) = (x & (x - 1)) == 0

  def permute_seq [n] (x: *[n]pf.mus.t) : *[n]pf.mus.t =
    let t = assert (is_power_of_2 n) (ceilLg n) in
    loop (x) for k < i32.i64 n do
      let j = bitReverse k t
      in if j > k
         then let x_j = copy x[j]
              let x_k = copy x[k]
              let x[j] = x_k
              let x[k] = x_j
              in  x
         else x

  def fft2_seq [n] (omega: pf.mus.t) (x: [n]pf.mus.t) : [n]pf.mus.t =
    let lgn = ceilLg n
    let x = permute_seq (copy x)
    in
    loop (x) for qm1 < lgn do
      let q = qm1 + 1
      let L = 1 << q
      let r = (i32.i64 n) >> q
      in
      loop (x) for k < r do
        let omega_pow = pf.us1
        let omega_step = omega pf.^ (pf.us_i32 r)
        let (x, _) = 
          loop (x, omega_pow) for j < L/2 do
            let omega_pow = 
                  if j == 0 then omega_pow 
                  else omega_pow pf.* omega_step
            let kLj = k*L + j
            let tau = omega_pow pf.* (copy x[kLj + L/2])
            let x[kLj + L/2] = (copy x[kLj]) pf.- tau
            let x[kLj]       = (copy x[kLj]) pf.+ tau
            in  (x, omega_pow)
        in  x
          
  def permute_par [n] (x: [n]pf.mus.t) : *[n]pf.mus.t = #[unsafe]
    let t = assert (is_power_of_2 n) (ceilLg n) in
    let f (k: i32) = 
        let j = bitReverse k t in
        if  j > k
        then (i64.i32 j, #[unsafe] x[k], i64.i32 k, #[unsafe] x[j])
        else (       -1,         pf.us0,        -1,         pf.us0)
    
    let (i1s, v1s, i2s, v2s) =
        iota (n/2) |> map i32.i64 |> map f |> unzip4
    let x' = scatter (copy x) (i1s++i2s) (v1s++v2s)
    let (i1s, v1s, i2s, v2s) =
        iota (n/2) |> map (+(n/2)) |> map i32.i64 |> map f |> unzip4
    in scatter x' (i1s++i2s) (v1s++v2s)

  def fft2_par [n] (omega: pf.mus.t) (x: [n]pf.mus.t) : [n]pf.mus.t = #[unsafe]
    let omega_pows = map (\i -> if i==0 then pf.us1 else omega) (iota n) |> scan (pf.*) (pf.us1)
    let lgn = ceilLg n
    let x = permute_par x
    let x'= replicate n pf.us0
    let (y, y') =
      loop (x : *[n]pf.mus.t, x' : *[n]pf.mus.t)
      for qm1 < lgn do
        let q = qm1 + 1
        let L = 1i32 << q
        let Ld2 = 1i32 << qm1
        let r = (i32.i64 n) >> q

        let f x (i: i32) =
          let (k, j) = ( i >> qm1, i & (Ld2 - 1) )
          let kLj = (k * L) + j
          let omega_pow = omega_pows[r*j]
          let tau = omega_pow pf.* #[unsafe] x[kLj + Ld2]
          let x_kLj = #[unsafe] x[kLj]
          in  (i64.i32 kLj, x_kLj pf.+ tau, i64.i32 (kLj+Ld2), x_kLj pf.- tau)
        in
        if (qm1 & 1 == 0)
        then let (is1, vs1, is2, vs2) =
                iota (n/2) |> map i32.i64 |> map (f x) |> unzip4
             let x' = scatter x' (is1++is2) (vs1++vs2)
             in  (x, x')
        else let (is1, vs1, is2, vs2) =
                iota (n/2) |> map i32.i64 |> map (f x') |> unzip4
             let x = scatter x (is1++is2) (vs1++vs2)
             in  (x, x')
    let (y1, y2) = iota (n/2) |>
        map (\ i -> if (lgn & 1) == 1 
                    then #[unsafe] ( y[i],  y[i+n/2])
                    else #[unsafe] (y'[i], y'[i+n/2])
            ) |> unzip
    in  ((y1 ++ y2) :> *[n]pf.mus.t) |> opaque

  def fft2_par_bad [n] (omega: pf.mus.t) (x: [n]pf.mus.t) : *[n]pf.mus.t = #[unsafe]
    let omega_pows = map (\i -> if i==0 then pf.us1 else omega) (iota n) |> scan (pf.*) (pf.us1)
    let lgn = ceilLg n
    let x = permute_par x
    let x'= replicate n pf.us0
    let (y,_) =
      loop (x : *[n]pf.mus.t, x' : *[n]pf.mus.t)
      for qm1 < lgn do
        let q = qm1 + 1
        let L = 1i32 << q
        let Ld2 = 1i32 << qm1
        let r = (i32.i64 n) >> q

        let f x (i: i32) =
          let (k, j) = ( i >> qm1, i & (Ld2 - 1) )
          let kLj = (k * L) + j
          let omega_pow = omega_pows[r*j]
          let tau = omega_pow pf.* #[unsafe] x[kLj + Ld2]
          let x_kLj = #[unsafe] x[kLj]
          in  (i64.i32 kLj, x_kLj pf.+ tau, i64.i32 (kLj+Ld2), x_kLj pf.- tau)
        
        let (is1, vs1, is2, vs2) =
                iota (n/2) |> map i32.i64 |> map (f x) |> unzip4
        let x' = scatter x' (is1++is2) (vs1++vs2)
        in  (x', x)
    let (y1, y2) = iota (n/2) |>
        map (\ i -> (#[unsafe] y[opaque i], #[unsafe] y[opaque i+n/2]) ) |> unzip2
    in  ((y1 ++ y2) :> *[n]pf.mus.t) |> opaque


  def ffft2 [n] (x : [n]pf.mus.t) : [n]pf.mus.t =
    fft2_par (getOmega n) x
    
  def ifft2 [n] (x : [n]pf.mus.t) : [n]pf.mus.t =
    let omega_inv = getOmega n |> pf.inv
    let n_inv = i32.i64 n |> pf.us_i32 |> pf.inv
    let r = fft2_par omega_inv x
    let r1 = map (pf.* n_inv) (r[0:n/2])
    let r2 = map (pf.* n_inv) (r[n/2:n])
    in  (r1++r2) :> [n]pf.mus.t
     
  def fftmul [n] (x : [n]pf.mus.t) (y : [n]pf.mus.t) : [n]pf.mus.t =
    let x' = ffft2 x |> opaque
    let y' = ffft2 y |> opaque
    let xy1 = map2 (pf.*) (x'[0:n/2]) (y'[0:n/2])
    let xy2 = map2 (pf.*) (x'[n/2:n]) (y'[n/2:n])
    let xy' = xy1 ++ xy2
    in  ifft2 (xy' :> [n]pf.mus.t)
}

module zmod32 = mk_zmod uz32
module fftu32 = mk_fftz1d zmod32

entry main (a: u32) (b: u32) : u32 =
  (zmod32.per a) zmod32.* (zmod32.per b) |> zmod32.rep
  
  
-- Testing FFT
-- ==

-- entry: testFFTseq testFFTpar
-- compiled input  { [ 11400u32, 28374u32, 23152u32, 9576u32
--                   , 29511u32, 20787u32, 13067u32, 14015u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   , 0u32, 0u32, 0u32, 0u32 
--                   ] }
--          output { [ 149882u32, 2275842870u32, 1182190460u32, 2383281722u32
--                   , 1032368839u32, 2591113531u32, 1041148619u32, 1666456763u32
--                   , 4378u32, 379675358u32, 2299045438u32, 2902395118u32
--                   , 2188866018u32, 1972597972u32, 1919993985u32, 1934855231u32
--                   ]
--                 } 
-- compiled input  { [ 30268u32, 20788u32, 8033u32, 15446u32
--                   , 26275u32, 11619u32,  2494u32,  7016u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   ] 
--                 }
--          output { [ 121939u32, 1466303512u32, 2586179827u32, 2215991939u32
--                   , 1959268324u32, 3184250004u32, 1699548738u32, 3026849220u32
--                   , 12201u32, 3020035466u32, 549183621u32, 389018726u32
--                   , 1262049181u32, 799363395u32, 1607554732u32, 2004557247u32
--                   ]
--                 }

def testFFTseq [n] (inp: [n]u32) : [n]u32 =
  map fftu32.elper inp |> fftu32.fft2_seq (fftu32.getOmega n) |> map fftu32.elrep

def testFFTpar [n] (inp: [n]u32) : [n]u32 =
  map fftu32.elper inp |> fftu32.ffft2 |> map fftu32.elrep
  
-- Testing FFT-MUL (without final carry step)
-- ==

-- entry: testFFTmul
-- compiled input  { [ 11400u32, 28374u32, 23152u32, 9576u32
--                   , 29511u32, 20787u32, 13067u32, 14015u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   , 0u32, 0u32, 0u32, 0u32 
--                   ]
--                   [ 30268u32, 20788u32, 8033u32, 15446u32
--                   , 26275u32, 11619u32,  2494u32,  7016u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   , 0u32, 0u32, 0u32, 0u32
--                   ]
--                 }
--          output { [ 345055200u32, 1095807432u32, 1382179648u32, 1175142886u32
--                   , 2016084656u32, 2555168834u32, 2179032777u32, 1990011337u32
--                   , 1860865174u32, 1389799087u32, 942120918u32, 778961552u32
--                   , 341270975u32, 126631482u32, 98329240u32, 0u32
--                   ]
--                 }
def testFFTmul [n] (x1: [n]u32) (x2: [n]u32) : [n]u32 =
  fftu32.fftmul (map fftu32.elper x1) (map fftu32.elper x2)
  |> map fftu32.elrep
  
  
-- Big-Integer Multiplication: performance
-- ==
-- entry:  perfFFToneMul poly
-- compiled random input {  [65536][4096]u32   [65536][4096]u32 }
-- compiled random input { [131072][2048]u32  [131072][2048]u32 }
-- compiled random input { [262144][1024]u32  [262144][1024]u32 }
-- compiled random input { [524288][512]u32   [524288][512]u32 }
-- compiled random input { [1048576][256]u32  [1048576][256]u32 }
-- compiled random input { [2097152][128]u32  [2097152][128]u32 }

let imap2Intra as bs f = #[incremental_flattening(only_intra)] map2 f as bs
let imapIntra  as f = #[incremental_flattening(only_intra)] map f as

entry perfFFToneMul [m][n] (x1s: [m][n]u32) (x2s: [m][n]u32) : [m][n]u32 = #[unsafe]
  let f x1 x2 = fftu32.fftmul (map fftu32.elper x1) (map fftu32.elper x2) 
             |> map fftu32.elrep
  in  imap2Intra x1s x2s f

entry perfFFToneMul0 [m][n] (x1s: [m][n]u32) (x2s: [m][n]u32) : [m][n]u32 = #[unsafe]
  let fft2 x = fftu32.ffft2 (map fftu32.elper x)
  let fmul x y = map2 (fftu32.pf.mus.*) x y
  let ift2 xy = fftu32.ffft2 xy |> map fftu32.elrep
  
  -- in imapIntra x1s fft2 |> map (map fftu32.elrep)
  
  let x1s' = imapIntra x1s fft2 |> opaque
  let x2s' = imapIntra x2s fft2 |> opaque
  let xys  = map2 (\x1 x2 -> fmul x1 x2) x1s' x2s' 
  let res  = imapIntra xys ift2
  in  res
  
-- computes: (a^2 + b) * (b^2 + b) + a*b
entry poly [m][n] (ass: [m][n]u32) (bss: [m][n]u32) : [m][n]u32 = 
  #[unsafe]
   imap2Intra ass bss
    (\ a b ->
        let a    = map fftu32.elper a
        let a2   = fftu32.fftmul a  a   -- a^2
        let b    = map fftu32.elper b
        let b2   = fftu32.fftmul b  b   -- b^2
        let ab2  = fftu32.fftmul a2 b2  -- a^2 * b^2
        let res  = fftu32.fftmul b2 ab2 -- a^2 * b^4
        -- let res = ab2
        in  map fftu32.elrep res
    ) 
