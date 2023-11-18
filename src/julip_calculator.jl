
import ACEpotentials.JuLIP: SitePotential, AtomicNumber
import ACEpotentials.JuLIP.Potentials: evaluate!, evaluate_d!, cutoff 

              

struct UFACE_JuLIP{TACE} <: SitePotential
   ace::TACE
end 


cutoff(jpot::UFACE_JuLIP) = jpot.ace.pairpot.rcut

function evaluate!(tmp, calc::UFACE_JuLIP, Rs, Zs, z0)
   return evaluate(calc.ace, Rs, Zs, z0) 
end

function evaluate_d!(dEs, tmp_d, calc::UFACE_JuLIP, Rs, Zs, z0)
   Ei, _dEs = evaluate_ed(calc.ace, Rs, Zs, z0) 
   for i = 1:length(Rs)
      dEs[i] = _dEs[i]
   end
   release!(_dEs)
   return dEs 
end


# --------------------------------------------------------
# experimental new calculators 

import ACEpotentials.JuLIP: Atoms, neighbourlist, maxneigs 
cutoff(ace::UFACE) = ace.pairpot.rcut

ACEpotentials.JuLIP.NeighbourLists._grow_array!(A::AbstractArray, args...) = nothing 

function energy_new(ace::UFACE, at::Atoms)
   TF = eltype(eltype(at.X))
   nlist = neighbourlist(at, cutoff(ace))
   maxneigs = ACEpotentials.JuLIP.maxneigs(nlist) 
   Rs = acquire!(ace.pool, :calc_Rs, (maxneigs,), SVector{3, TF})
   Zs = acquire!(ace.pool, :calc_Zs, (maxneigs,), AtomicNumber)
   tmp = (R = unwrap(Rs), Z = unwrap(Zs),)

   E = zero(TF)

   for i = 1:length(at) 
      Js, Rs, Zs = ACEpotentials.JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      z0 = at.Z[i] 
      E += evaluate(ace, Rs, Zs, z0)
   end

   release!(Rs)
   release!(Zs)

   return E 
end