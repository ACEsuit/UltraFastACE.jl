

#
# some auxiliary functions for UF_ACE evaluation 
#

# ------------------------------ 


"""
an array that setting an element doesn't change anything, 
while getting an element returns zero(T). This can be used as
an adjoint that we know we don't need. 
"""
struct BlackHole{T} end 

BlackHole(T) = BlackHole{T}()

Base.getindex(bh::BlackHole{T}, i...) where{T} = zero(T)

Base.setindex!(bh::BlackHole, v, i...) = v

Base.size(bh::BlackHole) = (Inf, Inf, Inf, Inf, Inf, Inf)
Base.size(bh::BlackHole, i::Integer) = Inf

Base.fill!(bh::BlackHole, args...) = bh 

Base.eltype(bh::BlackHole{T}) where {T} = T 

# ------------------------------ 
#  spherical harmonics 
#  these are some simple wrapper functions around SpheriCart 

import SpheriCart
import SpheriCart: SphericalHarmonics, compute!, 
                   compute_with_gradients! 

_get_L(ybasis::SphericalHarmonics{L}) where {L} = L 

_len_ylm(ybasis) = (_get_L(ybasis) + 1)^2

function evaluate_ylm(ace, Rs)
   TF = eltype(eltype(Rs))
   Zlm = zeros(TF, length(Rs), _len_ylm(ace.ybasis))
   evaluate_ylm!(Zlm, ace, Rs)
   return Zlm 
end

function evaluate_ylm!(Zlm, ace, Rs)
   compute!(Zlm, ace.ybasis, Rs)
   return Zlm 
end

function whatalloc(::typeof(evaluate_ylm!), ace, Rs)
   TF = eltype(eltype(Rs))
   return (TF, length(Rs), _len_ylm(ace.ybasis))
end

function evaluate_ylm_ed(ace, Rs)
   TF = eltype(eltype(Rs))
   Zlm = zeros(TF, (length(Rs), _len_ylm(ace.ybasis)))
   dZlm = zeros(SVector{3, TF}, (length(Rs), _len_ylm(ace.ybasis)))
   compute_with_gradients!(Zlm, dZlm, ace.ybasis, Rs)
   return Zlm, dZlm
end

function evaluate_ylm_ed!(Zlm, dZlm, ace, Rs)
   compute_with_gradients!(Zlm, dZlm, ace.ybasis, Rs)
   return Zlm, dZlm
end

function whatalloc(::typeof(evaluate_ylm_ed!), ace, Rs)
   TF = eltype(eltype(Rs))
   return (TF, length(Rs), _len_ylm(ace.ybasis)), 
          (SVector{3, TF}, length(Rs), _len_ylm(ace.ybasis))
end



# ------------------------------ 
#  element embedding 

function embed_z(ace, Rs, Zs)
   TF = eltype(eltype(Rs))
   Ez = zeros(TF, length(Zs), length(ace.rbasis))
   return embed_z!(Ez, ace, Rs, Zs)
end


function embed_z!(Ez, ace, Rs, Zs)
   fill!(Ez, 0)
   for (j, z) in enumerate(Zs)
      iz = _z2i(ace.rbasis, z)
      Ez[j, iz] = 1
   end
   return Ez 
end


function whatalloc(::typeof(embed_z!), ace, Rs, Zs)
   TF = eltype(eltype(Rs))
   return (TF, length(Zs), length(ace.rbasis))
end


# ------------------------------ 
#  aadot via P4ML 

using LinearAlgebra: dot 

struct AADot{T, TAA}
   cc::Vector{T} 
   aabasis::TAA
end

function (aadot::AADot)(A)
   @no_escape begin 
      AA = @alloc(eltype(A), length(aadot.aabasis))
      P4ML.evaluate!(AA, aadot.aabasis, A)
      out = dot(aadot.cc, AA)
   end
   return out 
end

function eval_and_grad!(∇φ_A, aadot::AADot, A)
   φ = aadot(A)
   P4ML.pullback!(∇φ_A, aadot.cc, aadot.aabasis, A)
   return φ
end