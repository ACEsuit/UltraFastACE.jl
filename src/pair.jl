
using ACEpotentials.ACE1.PairPotentials: PolyPairPot
using ACEpotentials.ACE1: cutoff

struct SplinePairPot{NZ, T, SPL}
   _i2z::NTuple{NZ, Int}
   spl::SPL   
   rcut::T 
   rin::T 
end


function make_pairpot_splines(pairpot; n_spl_points = 10_000, rin = 1e-10)
   @assert pairpot isa PolyPairPot
   zlist = pairpot.basis.zlist.list
   rcut = cutoff(pairpot)
   rspl = range(rin, rcut; length = n_spl_points)
   function _make_pair_spl(z1, z0)
      yspl = [ evaluate(pairpot, r, z1, z0) for r in rspl ]
      return CubicSplineInterpolation(rspl, yspl)
   end
   spl = Dict([ (z1, z0) => _make_pair_spl(z1, z0) 
                 for z0 in zlist for z1 in zlist ])
   _i2z = Int.(tuple(zlist...))
   return SplinePairPot(_i2z, spl, rcut, rin)
end


function evaluate(pot::SplinePairPot, r::Number, z1, z0)
   if r < pot.rin 
      return Inf 
   end 
   if r > pot.rcut
      return 0.0 
   end

   return pot.spl[(z1, z0)](r)
end

function evaluate_d(pot::SplinePairPot, r::Number, z1, z0)
   if r < pot.rin 
      return Inf 
   end 
   if r > pot.rcut
      return 0.0 
   end

   return Interpolations.gradient1(pot.spl[(z1, z0)], r)
end


function evaluate(pot::SplinePairPot, Rs::AbstractVector, Zs::AbstractVector, z0)
   TF = eltype(eltype(Rs))                  
   v = zero(TF)
   for ij = 1:length(Rs) 
      v += evaluate(pot, norm(Rs[ij]), Zs[ij], z0)
   end
   return v / 2
end

function add_evaluate_d!(dEs, pot::SplinePairPot, Rs::AbstractVector, Zs::AbstractVector, z0)
   TF = eltype(eltype(Rs))                  
   for ij = 1:length(Rs) 
      rij = norm(Rs[ij])
      dv = evaluate_d(pot, rij, Zs[ij], z0)
      dEs[ij] += (0.5*dv/rij) * Rs[ij]
   end
   return nothing 
end