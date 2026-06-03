import "badd"
import "classic-mul"

def poly [m][ipb][n][q] (ass: [m][ipb*n][2*q]u64) (bss: [m][ipb*n][2*q]u64) : [m][ipb*n][2*q]u64 = 
  #[unsafe]
  let a2pbs = opaque <|
              imap2Intra ass bss                         -- a^2 + b
               (\a b -> let a2 = bmul a a in badd a2 b)

  let b2pbs = opaque <|
              imapIntra bss                              -- b^2 + b
               (\b   -> let b2 = bmul b b in badd b2 b)

  let prods = opaque <|
              imap2Intra a2pbs b2pbs                     -- (a^2 + b) * (b^2 + b)
               (\a2pb b2pb -> bmul a2pb b2pb)

  let res   = opaque <|
              imap3Intra ass bss prods                   -- (a^2 + b) * (b^2 + b) + a*b
               (\ a b prod -> let ab = bmul a b in badd prod ab)

  in  res

