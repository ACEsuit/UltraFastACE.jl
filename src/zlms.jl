#
# This provides generated real solid harmonics (and spherical harmonics) 
# using the approach proposed by 
#       Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, 
#       Journal of Computer Graphics Techniques, Vol. 2, No. 2, 2013
# The ideas are expanded and details are worked out more nicely in 
#       Fast evaluation of spherical harmonics with sphericart, 
#       Filippo Bigi, Guillaume Fraux, Nicholas J. Browning and Michele Ceriotti
#       arXiv:2302.08381; J. Chem. Phys. 159, 064802 (2023)
#

using StaticArrays, OffsetArrays, StaticPolynomials
using DynamicPolynomials: @polyvar

"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1) * (maxL + 1)

"""
`lm2idx(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`. `Y_lm` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `lm2idx`: given an index into a vector of Ylm values, return the 
`l, m` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 


function _gen_Flm(L::Integer)
   Flm = OffsetMatrix(zeros(L+1, L+1), (-1, -1))
   for l = big(0):big(L)
      for m = big(0):big(l)
         Flm[l, m] = (-1)^m * sqrt( (2*l+1)/(2*π) * 
                                     factorial(l-m) / factorial(l+m) )
      end
   end
   return Flm
end



function _Zlms(L::Integer, rr::AbstractVector{T}) where {T}
   @assert length(rr) == 3
   Flm = _gen_Flm(L)
   len = sizeY(L)
   rt2 = sqrt(2) 

   s = OffsetVector(Vector{Any}(undef, L+1), (-1,))
   c = OffsetVector(Vector{Any}(undef, L+1), (-1,))
   Q = OffsetMatrix(Matrix{Any}(undef, (L+1, L+1)), (-1, -1))
   Z = OffsetMatrix(Matrix{Any}(undef, (L+1, 2*L+1)), (-1, -L-1))
   
   x = rr[1]   
   y = rr[2]
   z = rr[3]
   r² = x^2 + y^2 + z^2
   
   # c_m and s_m 
   s[0] = zero(T)
   c[0] = one(T) 
   for m = 1:L 
      s[m] = s[m-1] * x + c[m-1] * y
      c[m] = c[m-1] * x - s[m-1] * y
   end
   # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
   
   Q[0,0] = one(T)
   Z[0,0] = (Flm[0,0]/rt2) * Q[0,0]

   for l = 1:L 
      # Q_l^l and Y_l^l
      # m = l 
      Q[l,l] = - (2*l-1) * Q[l-1,l-1]
      Z[l,l] = Flm[l,l] * Q[l,l] * c[l]
      Z[l,-l] = Flm[l,l] * Q[l,l] * s[l]
      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      Q[l,l-1] = (2*l-1) * z * Q[l-1, l-1]
      Z[l,-l+1] = Flm[l,l-1] * Q[l,l-1] * s[l-1]
      Z[l,l-1] = Flm[l,l-1] * Q[l,l-1] * c[l-1]  # overwrite if m = 0 -> ok 
      # now we can go to the second recursion 
      for m = l-2:-1:0 
         Q[l,m] = (2*l-1) * z * Q[l-1,m] - (l+m-1) * r² * Q[l-2,m]
         Z[l,-m] = Flm[l,m] * Q[l,m] * s[m]
         Z[l,m] = Flm[l,m] * Q[l,m] * c[m]
      end
   end

   # finally generate an svector output 
   return [ getindex(Z, idx2lm(i)...) for i = 1:len ]
end


function generate_Zlms(L::Integer)
   @polyvar rr[1:3]  
   dynamic_system = _Zlms(L, rr)
   # a little trick to convert some terms to actual polynomials 
   dynamic_system = dynamic_system .+ (0*rr[1])
   # convert to a static system 
   return PolynomialSystem(dynamic_system)
end



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     BACKUP 
# The code below is a start of an explicitly generated code that 
# might be faster, and an approach like that might in particular 
# be faster for backpropagation. 

function _codegen_Zlm(L, T) 
   Flm = _gen_Flm(L)
   len = sizeY(L)
   rt2 = sqrt(2) 

   code = Expr[] 
   push!(code, :(x = rr[1]))
   push!(code, :(y = rr[2]))
   push!(code, :(z = rr[3]))
   push!(code, :(r² = x^2 + y^2 + z^2))

   # c_m and s_m 
   push!(code, :(s_0 = zero($T)))
   push!(code, :(c_0 = one($T)))
   for m = 1:L 
      push!(code, Meta.parse("s_$m = s_$(m-1) * x + c_$(m-1) * y"))
      push!(code, Meta.parse("c_$m = c_$(m-1) * x - s_$(m-1) * y"))
   end
   push!(code, Meta.parse("c_0 = one($T)/$rt2"))

   # Q_0^0 and Y_0^0
   push!(code, Meta.parse("Q_0_0 = one($T)"))
   push!(code, Meta.parse("Z_1 = $(Flm[0,0]/rt2) * Q_0_0"))

   for l = 1:L 
      # Q_l^l and Y_l^l
      # m = l 
      push!(code, Meta.parse("Q_$(l)_$(l)  = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l))  = $(Flm[l,l]) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l)) = $(Flm[l, l]) * Q_$(l)_$(l) * s_$(l)"))
      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      push!(code, Meta.parse("Q_$(l)_$(l-1)  = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l+1)) = $(Flm[l, l-1]) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l-1) ) = $(Flm[l, l-1]) * Q_$(l)_$(l-1) * c_$(l-1)" )) # overwrite if m = 0 -> ok 
      # now we can go to the second recursion 
      for m = l-2:-1:0 
         push!(code, Meta.parse("Q_$(l)_$(m)  = $(2*l-1) * z * Q_$(l-1)_$m - $(l+m-1) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,-m)) = $(Flm[l, m]) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,m) ) = $(Flm[l, m]) * Q_$(l)_$(m) * c_$(m)"))
      end
   end

   # finally generate an svector output
   push!(code, Meta.parse("return SVector{$len, $T}(" * 
                join( ["Z_$i, " for i = 1:len], ) * ")"))
end

@generated function Zlms(::Val{L}, rr::SVector{3, T}) where {L, T <: AbstractFloat}
   code = _codegen_Zlm(L, T)
   return quote
      $(Expr(:block, code...))
   end
end
