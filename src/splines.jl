
struct SparseStaticArray{N, T}
   idx::UnitRange{Int}
   data::SVector{N, T}
end


struct SplineRadialsZ{SPL, N}
   _i2z::NTuple{N, Int}
   idx::NTuple{N, Int}
   spl::NTuple{N, SPL}
end

struct SplineRadials{SPL, N}
   _i2z::NTuple{N, Int}
   spl::NTuple{N, SPL}
end


function evaluate(basis::SplineRadials, rij, Zj)
   i_Zj = _z2i(basis, Zj)
   spl_j = basis.spl[i_Zj] 
   return spl_j(rij)
end

function evaluate(basis::SplineRadialsZ, rij, Zj)
   i_Zj = _z2i(basis, Zj)
   spl_j = basis.spl[i_Zj] 
   return SparseStaticArray(basis.idx[i_Zj], spl_j(rij))
end

function evaluate!(out, basis::SplineRadials, Rs, Zs)
   @inbounds for ij = 1:length(Rs) 
      rij = norm(Rs[ij]) 
      zj = Zs[ij]
      i_zj = _z2i(basis, zj)
      spl_ij = basis.spl[i_zj] 
      out[ij, :] .= spl_j(rij)
      return out
   end
end