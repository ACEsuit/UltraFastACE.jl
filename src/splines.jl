
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

# Rn = acquire!(ace.pool, :Rn, (length(Rs), length(rbasis)), TF)
# evaluate!(Rn, rbasis, Rs, Zs)

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

