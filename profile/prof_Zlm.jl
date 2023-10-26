
using UltraFastACE, StaticArrays, BenchmarkTools, Test
using UltraFastACE: generate_Zlms, Zlms, idx2lm
using StaticPolynomials: jacobian 
using Polynomials4ML: RRlmBasis, evaluate, release! 
using ACEbase.Testing: print_tf, println_slim 
using StrideArrays: PtrArray
using LoopVectorization
import Polynomials4ML

##

L = 10
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


function Zlm_gen_N_alt!(Z, ZZ, Zlm, XX)
   broadcast!(Zlm, ZZ, XX)
   nX = length(XX)
   LEN = length(ZZ[1])
   @inbounds for a = 1:LEN
      @simd ivdep for i = 1:nX
         Z[i, a] = ZZ[i][a]
      end
   end
   return nothing  
end


@info("Batching a generated implementation")
@btime Zlm_gen_N!($Z, $Zlm_gen, $XX)


# 1.267 µs  = 1280 / 45 = 28.4 ns per evaluation
#     whereas the pure evaluation time is 18ns. 

@info("batching via broadcast (bad memory layout for pooling)")
ZZ = Zlm_gen.(XX)
@btime broadcast!($Zlm_gen, $ZZ, $XX)

# @btime Zlm_gen_N_alt!($Z, $ZZ, $Zlm_gen, $XX)


@info("P4ML implementation")
@btime Polynomials4ML.evaluate!($Z, $Zlm_p4ml, $XX)

# the analogous P4ML implementation is 2.167 ns. 
# this suggests that we should just write simd sphericart version. 

## -------------- 

@info("batching via SIMD (good memory layout for pooling)")
uf_Zlm = UltraFastACE.ZlmBasis(L)
UltraFastACE.evaluate!(Z, uf_Zlm, XX)

@btime UltraFastACE.evaluate!($Z, $uf_Zlm, $XX)

## -------------- 

@profview let Z = Z, uf_Zlm = uf_Zlm, XX = XX
   for n = 1:3_000_000
      UltraFastACE.evaluate!(Z, uf_Zlm, XX)
   end
end