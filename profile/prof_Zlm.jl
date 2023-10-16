
using UltraFastACE, StaticArrays, BenchmarkTools, Test
using UltraFastACE: generate_Zlms, Zlms, idx2lm
using StaticPolynomials: jacobian 
using Polynomials4ML: RRlmBasis, evaluate 
using ACEbase.Testing: print_tf, println_slim 

##

L = 5
Zlm_poly = generate_Zlms(L)
Zlm_p4ml = RRlmBasis(L)
Zlm_gen = let valL = Val(L); xx -> Zlms(valL, xx); end 

for ntest = 1:20 
   xx = @SVector randn(3) 
   Z_p = Zlm_poly(xx)
   Z_g = Zlm_gen(xx)
   print_tf(@test Z_p ≈ Z_g)
end
println()

##

xx0 = @SVector randn(3) 
xx1 = @SVector randn(3)
Z0_p = Zlm_poly(xx0)
Z0_4 = Zlm_p4ml(xx0)
Z1_p = Zlm_poly(xx1)
Z1_4 = Zlm_p4ml(xx1)

F_p = Z0_p ./ Z0_4 

display( [ Z0_4 F_p ])

display(
      [ (Z1_p - Z1_4 .* F_p) F_p ]
   )


##

@btime ($Z5)($xx)
@btime ($Z5)($xx)
@btime jacobian($Z5, $xx)
@btime jacobian($Z5, $xx)


##



@btime Zlms($valL, $xx)
@btime ($Z5)($xx)

Zlms(val3, xx)
@btime Zlms($val3, $xx)