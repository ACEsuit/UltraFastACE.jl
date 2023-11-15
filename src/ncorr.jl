

using StaticArrays, OffsetArrays
using StaticPolynomials: Polynomial, evaluate_and_gradient
using DynamicPolynomials: @polyvar

"""
This naive code is not supposed to be fast, it is only used to generate a 
dynamic polynomial representating the operation AA ⋅ c -> εᵢ 

The generated (giant) polynomial is then used to generate optimized 
evaluation and gradient code. 
"""
function _AA_dot(A, spec, c)
   T = promote_type(eltype(A), eltype(c))
   out = zero(T)
   for (i, kk) in enumerate(spec)
      out += c[i] * prod(A[kk[t]] for t = 1:length(spec[i]))
   end
   return out 
end


function generate_AA_dot(spec, c)
   nA = maximum(maximum, spec)
   @polyvar A[1:nA]
   dynamic_poly = _AA_dot(A, spec, c)
   return Polynomial(dynamic_poly)
end


