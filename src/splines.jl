
struct SparseStaticArray{N, T}
   idx::UnitRange{Int}
   data::SVector{N, T}
end


struct SplineRadialsZ{SPL, N, LEN}
   _i2z::NTuple{N, Int}
   spl::NTuple{N, SPL}
end

SplineRadialsZ(_i2z::NTuple{N, Int}, spl::NTuple{N, SPL}, LEN
              ) where {N, SPL} = 
         SplineRadialsZ{SPL, N, LEN}(_i2z, spl)

Base.length(basis::SplineRadialsZ{SPL, N, LEN}) where {SPL, N, LEN} = LEN

struct SplineRadials{SPL, N}
   _i2z::NTuple{N, Int}
   spl::NTuple{N, SPL}
end


function evaluate(ace, basis::SplineRadialsZ, 
                  Rs::AbstractVector{<: SVector}, Zs::AbstractVector)
   TF = eltype(eltype(Rs))                  
   Rn = acquire!(ace.pool, :Rn, (length(Rs), length(basis)), TF)
   evaluate!(Rn, basis, Rs, Zs)
   return Rn 
end


function evaluate!(out, basis::SplineRadialsZ, Rs, Zs)
   nX = length(Rs)
   len = length(basis)
   @assert length(Zs) >= nX
   @assert size(out, 1) >= nX
   @assert size(out, 2) >= len 

   @inbounds for ij = 1:nX 
      rij = norm(Rs[ij]) 
      zj = Zs[ij]
      i_zj = _z2i(basis, zj)
      spl_ij = basis.spl[i_zj] 
      out[ij, :] .= spl_ij(rij)
   end
   return out
end


function evaluate_ed(ace, rbasis, Rs, Zs)
   TF = eltype(eltype(Rs))                  
   Rn = acquire!(ace.pool, :Rn, (length(Rs), length(rbasis)), TF)
   dRn = acquire!(ace.pool, :dRn, (length(Rs), length(rbasis)), SVector{3, TF})
   evaluate_ed!(Rn, dRn, rbasis, Rs, Zs)
   return Rn, dRn 
end


function evaluate_ed!(Rn, dRn, basis::SplineRadialsZ, Rs, Zs)
   nX = length(Rs)
   len = length(basis)
   @assert length(Zs) >= nX
   @assert size(Rn, 1) >= nX
   @assert size(Rn, 2) >= len 
   @assert size(dRn, 1) >= nX
   @assert size(dRn, 2) >= len 

   for ij = 1:nX 
      rij = norm(Rs[ij]) 
      𝐫̂ij = Rs[ij] / rij 
      zj = Zs[ij]
      i_zj = _z2i(basis, zj)
      spl_ij = basis.spl[i_zj] 
      Rn[ij, :] .= spl_ij(rij)
      g = Interpolations.gradient1(spl_ij, rij)
      for n = 1:length(g) 
         dRn[ij, n] = g[n] * 𝐫̂ij
      end
   end
   return nothing 
end

