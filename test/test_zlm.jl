
using Test, StaticArrays, LinearAlgebra, Random
import UltraFastACE 
using UltraFastACE.SpheriCart

using ACEbase: evaluate, evaluate!
using ACEbase.Testing: print_tf, println_slim
using UltraFastACE.SpheriCart: sizeY 

using Polynomials4ML: RRlmBasis

using BenchmarkTools

##

# This is an implementation that ignores any normalisation factors. 
# Correctness of the implementation will be tested UP TO normalisation. 
# The normalisation will then separately be tested by computing the gramian 
# and confirming that the basis if L2-orthonormal on the sphere. 

# TODO: can we replace this against a generated code? (sympy or similar?)

function symbolic_zlm_4(𝐫)
   x, y, z = tuple(𝐫...)
   r = norm(𝐫)   
   [ 
      1.0,  # l = 0
      y,    # l = 1
      z, 
      x, 
      x * y,  # l = 2 
      y * z, 
      3 * z^2 - r^2,
      x * z, 
      x^2 - y^2, 
      (3 * x^2 - y^2) * y,   # l = 3
      x * y * z, 
      (5 * z^2 - r^2) * y, 
      (5 * z^2 - 3 * r^2) * z,
      (5 * z^2 - r^2) * x,
      (x^2 - y^2) * z, 
      (x^2 - 3 * y^2) * x, 
      x * y * (x^2 - y^2),    # l = 4 
      y * z * (3 * x^2 - y^2), 
      x * y * (7 * z^2 - r^2), 
      y * z * (7 * z^2 - 3 * r^2), 
      (35 * z^4 - 30 * r^2 * z^2 + 3 * r^4),
      x * z * (7 * z^2 - 3 * r^2),
      (x^2 - y^2) * (7 * z^2 - r^2),
      x * z * (x^2 - 3 * y^2),
      x^2 * (x^2 - 3 * y^2) - y^2 * (3 * x^2 - y^2),
   ]
end

# the code to be tested against the symbolic 
# the other implementations will be tested against this. 
zlm_4(𝐫) = static_solid_harmonics(Val(4), 𝐫)

𝐫0 = @SVector randn(3)
Z1 = zlm_4(𝐫0)
Z2 = symbolic_zlm_4(𝐫0)
F = Z1 ./ Z2

for ntest = 1:30 
   𝐫 = @SVector randn(3)
   Z1 = zlm_4(𝐫)
   Z2 = symbolic_zlm_4(𝐫)
   print_tf(@test Z1 ≈ Z2 .* F)
end

##

@info("confirm that the two implementations are consistent with one another")
for L = 2:10, ntest = 1:10
   basis = ZlmBasis(L)
   𝐫 = @SVector randn(3)
   Z1 = static_solid_harmonics(Val(L), 𝐫)
   Z2 = evaluate(basis, [𝐫,])[:]
   print_tf(@test Z1 ≈ Z2)
end
println()


##

@info("test the orthogonality on the sphere: G ≈ I")

Random.seed!(0)
L = 3
basis = ZlmBasis(L)
rand_sphere() = ( (𝐫 = @SVector randn(3)); 𝐫/norm(𝐫) )

for ntest = 1:10
   rr = [ rand_sphere() for _ = 1:10_000 ] 
   Z = evaluate(basis, rr)
   G = (Z' * Z) / length(rr) * 4 * π
   print_tf(@test norm(G - I) < 0.33) 
   print_tf(@test cond(G) < 1.5)
end
# println_slim(@test round.(G, digits=2) == I)


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

