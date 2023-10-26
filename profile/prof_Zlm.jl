
using UltraFastACE, StaticArrays, BenchmarkTools, Test
using UltraFastACE: generate_Zlms, Zlms, idx2lm
using StaticPolynomials: jacobian 
using Polynomials4ML: RRlmBasis, evaluate, release! 
using ACEbase.Testing: print_tf, println_slim 
using StrideArrays: PtrArray
using LoopVectorization
import Polynomials4ML

##

L = 3
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


@btime ($Zlm_gen)($xx0)
@btime ($Zlm_poly)($xx0)
@btime (Y = ($Zlm_p4ml)($xx0); release!(Y))


# @btime jacobian($Zlm_gen, $xx0)
# @btime jacobian($Zlm_gen, $xx0)


##

XX = randn(SVector{3, Float64}, 45)
nX = length(XX)
lenZ = length(Z0_p)
_Z = zeros(nX, lenZ)
Z = PtrArray(_Z)

function Zlm_gen_N!(Z, Zlm, XX)
   nX = length(XX) 
   @inbounds begin   
      @simd ivdep for i = 1:nX
         Z[i, :] .= Zlm(XX[i])
      end
   end
   return nothing  
end


@btime Zlm_gen_N!($Z, $Zlm_gen, $XX)
# 1.267 µs  = 1280 / 45 = 28.4 ns per evaluation
#     whereas the pure evaluation time is 18ns. 

@btime Polynomials4ML.evaluate!($Z, $Zlm_p4ml, $XX)

# the analogous P4ML implementation is 2.167 ns. 
# this suggests that we should just write simd sphericart version. 

## -------------- 

uf_Zlm = UltraFastACE.ZlmBasis(L)
UltraFastACE.evaluate!(Z, uf_Zlm, XX)

@btime UltraFastACE.evaluate!($Z, $uf_Zlm, $XX)
