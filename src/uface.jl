import Polynomials4ML
import ACEpotentials
using Interpolations
import SpheriCart
import ACEpotentials.ACE1
import ACEpotentials.ACE1: AtomicNumber
# using SpheriCart: SphericalHarmonics, compute

C2R = ConvertC2R
P4ML = Polynomials4ML

struct UFACE_inner{TR, TY, TA, TAA}
   rbasis::TR
   ybasis::TY
   abasis::TA
   aadot::TAA
   meta::Dict
end

struct UFACE{N, TR, TY, TA, TAA}
   _i2z::NTuple{N, Int}
   ace_inner::NTuple{N, UFACE_inner{TR, TY, TA, TAA}}
end


function evaluate(ace::UFACE, Rs, Zs, zi) 
   i_zi = _z2i(ace, zi)
   ace_inner = ace.ace_inner[i_zi]
   return evaluate(ace_inner, Rs, Zs)
end


function evaluate(ace::UFACE_inner, Rs, Zs)
   

end


# ------------------------------------------------------
# transformation code 

function make_radial_splines(Rn_basis, zlist; npoints = 100)
   @assert Rn_basis.envelope isa ACEpotentials.ACE1.OrthPolys.OneEnvelope
   rcut = Rn_basis.ru 
   rspl = range(0.0, rcut, length = npoints)
   function _make_rad_spl(z1, z0)
      yspl = [ SVector(ACE1.evaluate(Rn_basis, r, z1, z0)...) for r in rspl ]
      return CubicSplineInterpolation(rspl, yspl)
   end
   spl = Dict([ (z1, z0) => _make_rad_spl(z1, z0) for z0 in zlist, z1 in zlist ])
   return spl  
end

function make_ace1_AA_spec(mbpot, iz) 
   D_spec = mbpot.pibasis.inner[iz].b2iAA
   spec_c = Vector{Vector{NamedTuple{(:z, :n, :l, :m), Tuple{AtomicNumber, Int, Int, Int}}}}(undef, length(D_spec))
   for (bb1, idx) in D_spec 
      bb = [ (z = b.z, n = b.n, l = b.l, m = b.m) for b in bb1.oneps ]
      spec_c[idx] = bb
   end  
   return spec_c  
end

function make_AA_transform(mbpot, iz, D_T_Ylm)
   spec_AA_ace1 = make_ace1_AA_spec(mbpot, iz)
   return C2R._AA_r2c_transform(spec_AA_ace1, D_T_Ylm; f = real)
end



function uface_from_ace1_inner(mbpot, iz; n_spl_points = 100)
   b1p = mbpot.pibasis.basis1p
   zlist = b1p.zlist.list 
   z0 = zlist[iz]
   t_zlist = tuple(zlist...)

   # radial embedding
   Rn_basis = mbpot.pibasis.basis1p.J
   spl = make_radial_splines(Rn_basis, zlist; npoints = n_spl_points)
   rbasis_new = SplineRadialsZ(Int.(t_zlist), 
                     ntuple(iz1 -> spl[(zlist[iz1], z0)], length(zlist)))
   # P4ML style spec of radial embedding 
   spec2i_Rn = 1:length(Rn_basis.J)

   # P4ML style spec of categorical embedding 
   zlist = b1p.zlist.list 
   spec2i_Ez = Dict([zlist[i] => i for i = 1:length(zlist)]...)

   # angular embedding 
   L = b1p.SH.maxL
   cYlm_basis_ace1 = b1p.SH 
   rYlm_basis_sc = SpheriCart.SphericalHarmonics(L)
   len_Y = (L+1)^2
   # P4ML style spec of angular embedding
   spec2i_Ylm = Dict([ (l = P4ML.idx2lm(i)[1], m = P4ML.idx2lm(i)[2]) => i  
                       for i = 1:len_Y ]...)

   # transformation from ACE1 to SpheriCart 
   T_Ylm = C2R.r2c_transform(L) * C2R.sc2p4_transform(L)
   D_T_Ylm = C2R._dict_Ylm_transform(T_Ylm)

   # generate the AA basis transformation information 
   # this produces a Dict containing various information from 
   # which we can build both the A and AA bases. 
   AA_transform = make_AA_transform(mbpot, iz, D_T_Ylm)

   # from the AA basis spec we can construct the real A basis 
   AA_spec_r = AA_transform[:spec_r]
   A_spec_r = sort(unique(reduce(vcat, AA_spec_r)))
   inv_spec_A = Dict{NamedTuple, Int}() 
   spec_A_inds = Vector{NTuple{3, Int}}(undef, length(A_spec_r))
   for (i, b) in enumerate(A_spec_r)
      i_Ez = spec2i_Ez[b.z]
      i_Rn = spec2i_Rn[b.n]
      i_Ylm = spec2i_Ylm[(l = b.l, m = b.m)]
      inv_spec_A[b] = i 
      spec_A_inds[i] = (i_Ez, i_Rn, i_Ylm)
   end
   A_basis = P4ML.PooledSparseProduct(spec_A_inds)

   # from AA_transform we can also construct the real AA basis 
   AA_spec_r = AA_transform[:spec_r]
   inv_AA_spec_r = AA_transform[:inv_spec_r]
   spec_AA_inds = Vector{Vector{Int}}(undef, length(AA_spec_r))
   for (i, bb) in enumerate(AA_spec_r)
      spec_AA_inds[i] = sort([ inv_spec_A[b] for b in bb ])
   end

   # AA_basis = P4ML.SparseSymmProd(spec_AA_inds)
   c_r_iz = AA_transform[:T]' * mbpot.coeffs[iz]
   aadot = generate_AA_dot(spec_AA_inds, c_r_iz)


   return UFACE_inner(rbasis_new, rYlm_basis_sc, A_basis, aadot, Dict()), AA_transform 
end
