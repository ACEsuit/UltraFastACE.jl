

_symmkey(z1::Integer, z0::Integer) = z1 < z0 ? (z1, z0) : (z0, z1)

struct SplineWithEnvelope{SPL, ENV} 
   spl::SPL
   env::ENV
end

mutable struct SymmPairBasis{BAS, TCUT} 
   bas::Dict{Tuple{Int, Int}, BAS}
   rcut::TCUT
   meta::Dict{String, Any}
   pool::TSafe{ArrayPool{FlexArrayCache}}
end


function evaluate(basis::SymmPairBasis, 
                  Rs::AbstractVector{<: SVector}, Zs::AbstractVector, z0)
   TF = eltype(eltype(Rs))
   Rn = acquire!(basis.pool, :Rn, (length(Rs), length(basis)), TF)
   evaluate!(Rn, basis, Rs, Zs)
   return Rn 
end
