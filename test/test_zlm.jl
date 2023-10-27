
using Test, StaticArrays
import UltraFastACE 
using UltraFastACE.SpheriCart

using ACEbase: evaluate, evaluate!
using ACEbase.Testing: print_tf
using UltraFastACE.SpheriCart: sizeY 

##

# function symbolic_zlm(L)   
# end



## 

@info("confirm that the two codes are consistent with one another")
for L = 2:10, ntest = 1:10
   basis = ZlmBasis(L)
   𝐫 = @SVector randn(3)
   Z1 = static_solid_harmonics(Val(L), 𝐫)
   Z2 = evaluate(basis, [𝐫,])[:]
   print_tf(@test Z1 ≈ Z2)
end
println()

##

@info("confirm batched evaluation is consistent with single")
for L = 2:10, ntest = 1:10
   basis = ZlmBasis(L)
   nbatch = rand(8:20)
   Rs = [ @SVector randn(3) for _=1:nbatch ]
   Z1 = reinterpret(reshape, Float64, 
                     static_solid_harmonics.(Val(L), Rs), )'
   Z2 = evaluate(basis, Rs)

   print_tf(@test Z1 ≈ Z2)
end
println()

##

