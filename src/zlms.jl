#
# This provides generated real solid harmonics (and spherical harmonics) 
# using the approach proposed by 
#       [1] Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, 
#       Journal of Computer Graphics Techniques, Vol. 2, No. 2, 2013
# The ideas are expanded on and details are worked out very nicely in 
#       [2] Fast evaluation of spherical harmonics with sphericart, 
#       Filippo Bigi, Guillaume Fraux, Nicholas J. Browning and Michele Ceriotti
#       arXiv:2302.08381; J. Chem. Phys. 159, 064802 (2023)
# Some aspects of this implementation, specifically how to deal with the 
# singularity, but applied in a different coordinate system were already 
# in ACE1.jl, but this implementation works purely with cartesian coordinates
# which is much more elegant. The performance seems comparable for large 
# bases (maxL large) but we gain about a factor 2-3 for small bases 
# in the range L = 3, 4, 5. 
#
# The current version of the code was written only with reference to 
# the published paper [2] and without consulting the reference implementation 
# provided by the authors. 
#
module SpheriCart

using StaticArrays, OffsetArrays, StaticPolynomials

"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_l^m`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1)^2

"""
`lm2idx(l,m):`
Return the index into a flat array of real spherical harmonics ``Y_l^m``
for the given indices `(l,m)`. ``Y_l^m`` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `lm2idx`: given an index into a vector of Ylm values, return the 
`(l, m)` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 

"""
Generates the `F[l, m]` values exactly as described in [2]. 
"""
function generate_Flms(L::Integer)
   Flm = OffsetMatrix(zeros(L+1, L+1), (-1, -1))
   for l = big(0):big(L)
      for m = big(0):big(l)
         Flm[l, m] = (-1)^m * sqrt( (2*l+1)/(2*π) * 
                                     factorial(l-m) / factorial(l+m) )
      end
   end
   return Flm
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Implementation 1: 
#  This is a generated code for a single input and maximal performance
#  (with a single input) 
#  

function _codegen_Zlm(L, T) 
   Flm = generate_Flms(L)
   len = sizeY(L)
   rt2 = sqrt(2) 

   code = Expr[] 
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

@generated function static_solid_harmonics(::Val{L}, x::T, y::T, z::T) where {L, T <: AbstractFloat}
   code = _codegen_Zlm(L, T)
   return quote
      $(Expr(:block, code...))
   end
end

zlms(valL::Val{L}, rr::SVector{3, T}) where {L, T} = 
      zlms(valL, rr[1], rr[2], rr[3])


# ------------------------------------ 

using ObjectPools
struct ZlmBasis{L, T1}  
   Flm::OffsetMatrix{T1, Matrix{T1}}
   cache::ArrayPool{FlexArrayCache}
end

function ZlmBasis(L::Integer) 
   Flm = generate_Flms(L)
   ZlmBasis{L, eltype(Flm)}(Flm, ArrayPool(FlexArrayCache))
end

function evaluate(basis::ZlmBasis{L}, 
                  Rs::AbstractVector{SVector{3, T}}) where {L, T}
   Z = zeros(length(Rs), sizeY(L))
   evaluate!(Z, basis, Rs)
   return Z
end

function evaluate!(Z::AbstractMatrix, 
                   basis::ZlmBasis{L}, 
                   Rs::AbstractVector{SVector{3, T}}) where {L, T} 

   nX = length(Rs)

   # get some temporary arrays for x, y, z coordinates from the cache
   x = acquire!(basis.cache, :x, (nX, ), T)
   y = acquire!(basis.cache, :y, (nX, ), T)
   z = acquire!(basis.cache, :z, (nX, ), T)

   @inbounds @simd ivdep for j = 1:nX
      rr = Rs[j] 
      xj, yj, zj = rr[1], rr[2], rr[3]
      x[j] = xj 
      y[j] = yj 
      z[j] = zj 
   end 

   # allocate temporary arrays from an array cache 
   temps = (r² = acquire!(basis.cache, :r2, (nX, ),   T),
            s = acquire!(basis.cache, :s,  (nX, L+1), T),
            c = acquire!(basis.cache, :c,  (nX, L+1), T),
            Q = acquire!(basis.cache, :Q,  (nX, sizeY(L)), T), 
            Flm = basis.Flm )

   # the actual evaluation kernel 
   solid_harmonics!(Z, Val{L}(), x, y, z, temps)

   # release the temporary arrays back into the cache
   release!(temps.x)
   release!(temps.y)
   release!(temps.z)
   release!(temps.r²)
   release!(temps.s)
   release!(temps.c)
   release!(temps.Q)

   return Z 
end 


# ----- Computational kernel for `ZlmBasis`

function solid_harmonics!(Z::AbstractMatrix, ::Val{L}, 
            x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, 
            temps::NamedTuple 
            ) where {L, T <: AbstractFloat}

   nX = length(x)
   len = sizeY(L)

   r² = temps.r²
   s = temps.s
   c = temps.c
   Q = temps.Q
   Flm = temps.Flm

   # some size checks to make sure the inbounds macro can be used safely. 
   @assert length(y) == length(z) == nX 
   @assert length(r²) >= nX
   @assert size(Z, 1) >= nX && size(s, 1) >= nX  && size(c, 1) >= nX && size(Q, 1) >= nX
   @assert size(Z, 2) >= len && size(Q, 2) >= len 
   @assert size(s, 2) >= L+1 && size(c, 2) >= L+1

   rt2 = sqrt(2) 
   
   @inbounds @simd ivdep for j = 1:nX
      r²[j] = x[j]^2 + y[j]^2 + z[j]^2
      # c_m and s_m, m = 0 
      s[j, 1] = zero(T)    # 0 -> 1
      c[j, 1] = one(T)     # 0 -> 1
   end

   # c_m and s_m continued 
   @inbounds for m = 1:L 
      @simd ivdep for j = 1:nX
         # m -> m+1 and  m-1 -> m
         s[j, m+1] = s[j, m] * x[j] + c[j, m] * y[j]
         c[j, m+1] = c[j, m] * x[j] - s[j, m] * y[j]
      end
   end

   # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
   i00 = lm2idx(0, 0)

   @inbounds @simd ivdep for j = 1:nX
      c[j, 1] = one(T)/rt2

      # fill Q_0^0 and Z_0^0 
      Q[j, i00] = one(T)
      Z[j, i00] = (Flm[0,0]/rt2) * Q[j, i00]
   end

   @inbounds for l = 1:L 
      ill = lm2idx(l, l)
      il⁻l = lm2idx(l, -l)
      ill⁻¹ = lm2idx(l, l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹ = lm2idx(l, -l+1)
      F_l_l = Flm[l,l]
      F_l_l⁻¹ = Flm[l,l-1]
      @simd ivdep for j = 1:nX 
         # Q_l^l and Y_l^l
         # m = l 
         Q[j, ill]   = - (2*l-1) * Q[j, il⁻¹l⁻¹]
         Z[j, ill]   = F_l_l * Q[j, ill] * c[j, l+1]  # l -> l+1
         Z[j, ill⁻¹] = F_l_l * Q[j, ill] * s[j, l+1]  # l -> l+1
         # Q_l^l-1 and Y_l^l-1
         # m = l-1 
         Q[j, ill⁻¹]  = (2*l-1) * z[j] * Q[j, il⁻¹l⁻¹]
         Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q[j, ill⁻¹] * s[j, l]  # l-1 -> l
         Z[j, ill⁻¹]  = F_l_l⁻¹ * Q[j, ill⁻¹] * c[j, l]  # l-1 -> l
         # overwrite if m = 0 -> ok 
      end

      # now we can go to the second recursion 
      for m = l-2:-1:0 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         F_l_m = Flm[l,m]
         @simd ivdep for j = 1:nX 
            Q[j, ilm] = (2*l-1) * z[j] * Q[j, il⁻¹m] - (l+m-1) * r²[j] * Q[j, il⁻²m]
            Z[j, il⁻m] = F_l_m * Q[j, ilm] * s[j, m+1]   # m -> m+1
            Z[j, ilm] = F_l_m * Q[j, ilm] * c[j, m+1]    # m -> m+1
         end
      end
   end

   return Z 
end
   
end 