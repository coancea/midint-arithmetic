import "badd"
import "classic-mul"

def poly [m][ipb][n][q] (ass: [m][ipb*n][2*q]u64) (bss: [m][ipb*n][2*q]u64) : [m][ipb*n][2*q]u64 = 
  #[unsafe]
  imap2Intra ass bss
    (\ a b ->
        let a2   = bmul a  a            -- a^2
        let a2pb = badd a2 b    -- a^2 + b
        let b2   = bmul b  b            -- b^2
        let b2pb = badd b2 b    -- b^2 + b
        let prod = bmul a2pb b2pb       -- (a^2 + b) * (b^2 + b)
        let ab   = bmul a  b            -- a*b
        let res  = badd prod ab -- (a^2 + b) * (b^2 + b) + a*b
        in  res
    )

