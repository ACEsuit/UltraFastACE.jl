import Polynomials4ML
using Interpolations
import ACE1
import ACE1: AtomicNumber, PIPotential, OneBody 
using LinearAlgebra: norm 
using StaticPolynomials: evaluate_and_gradient!


const C2R = ConvertC2R
const P4ML = Polynomials4ML

struct UFACE_inner{TR, TY, TA, TAA}
   rbasis::TR
   ybasis::TY
   abasis::TA
   aadot::TAA
   # ---------- admin and meta-data 
   # pool::TSafe{ArrayPool{FlexArrayCache}}
   meta::Dict
end

UFACE_inner(rbasis, ybasis, abasis, aadot) = 
   UFACE_inner(rbasis, ybasis, abasis, aadot, 
               # TSafe(ArrayPool(FlexArrayCache)), 
               Dict())

struct UFACE{NZ, INNER, PAIR}
   _i2z::NTuple{NZ, Int}
   ace_inner::INNER
   pairpot::PAIR
   E0s::Dict{Int, Float64}
   # ---------- 
   # pool::TSafe{ArrayPool{FlexArrayCache}}
   meta::Dict
end

UFACE(_i2z, ace_inner, pairpot, E0s) = 
      UFACE(_i2z, ace_inner, pairpot, E0s,
            # TSafe(ArrayPool(FlexArrayCache)), 
            Dict())




# ------------------------------------------------------
# transformation code : ACE1 -> UF_ACE models 

function make_radial_splines(Rn_basis, zlist; npoints = 100)
   @assert Rn_basis.envelope isa ACE1.OrthPolys.OneEnvelope
   rcut = Rn_basis.ru 
   rspl = range(0.0, rcut, length = npoints)
   function _make_rad_spl(z1, z0)
      yspl = [ SVector(ACE1.evaluate(Rn_basis, r, z1, z0)...) for r in rspl ]
      return cubic_spline_interpolation(rspl, yspl)
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



function uface_from_ace1_inner(mbpot, iz; 
                           n_spl_points = 100, 
                           aa_static = :auto)
   b1p = mbpot.pibasis.basis1p
   zlist = b1p.zlist.list 
   z0 = zlist[iz]
   t_zlist = tuple(zlist...)

   if aa_static == :auto 
      aa_static = (length(mbpot.pibasis.inner[1]) < 1_200)
   end

   # radial embedding
   Rn_basis = mbpot.pibasis.basis1p.J
   LEN_Rn = length(Rn_basis.J)
   spl = make_radial_splines(Rn_basis, zlist; npoints = n_spl_points)
   rbasis_new = SplineRadialsZ(Int.(t_zlist), 
                     ntuple(iz1 -> spl[(zlist[iz1], z0)], length(zlist)), 
                     LEN_Rn)
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

   if aa_static
      aadot = generate_AA_dot(spec_AA_inds, c_r_iz)
   else 
      aabasis = P4ML.SparseSymmProd(spec_AA_inds)
      aadot = AADot(c_r_iz, aabasis)
   end 

   return UFACE_inner(rbasis_new, rYlm_basis_sc, A_basis, aadot)
end




function uface_from_ace1(pot; n_spl_points = 100, 
                              n_spl_points_pair = 10_000, 
                              aa_static = :auto  )
   # generate the pair potential 
   pairpot = missing 
   for pc in pot.components 
      if pc isa PolyPairPot 
         @info("Importing pair potential model")
         pairpot = make_pairpot_splines(pc; n_spl_points = n_spl_points_pair)
         break 
      end
   end
   if ismissing(pairpot)
      @info("No pair potential found in ACE1 model")
   end

   # generate the many-body potential
   ace_inner = missing 
   _i2z = missing 
   for pc in pot.components 
      if pc isa PIPotential
         @info("Importing many-body potential")
         mbpot = pc 
         NZ = length(mbpot.pibasis.zlist)
         _i2z = tuple(Int.(mbpot.pibasis.zlist.list)...)
         ace_inner = tuple( 
            [ uface_from_ace1_inner(mbpot, iz; n_spl_points = n_spl_points, 
                                    aa_static = aa_static) 
              for iz = 1:NZ ]... )
         break 
      end
   end
   if ismissing(ace_inner)
      error("No many-body potential found in ACE1 model; I'm giving up.")
   end

   # 1-body potential 
   Eref = missing 
   for pc in pot.components 
      if pc isa OneBody
         @info("Importing 1-body potential")
         Eref = Dict([ Int(AtomicNumber(sym)) => e0 
                      for (sym, e0) in pc.E0 ]...)
         break 
      end
   end
   if ismissing(Eref)
      @info("No 1-body potential found in ACE1 model")
   end

   # return the UF_ACE model
   return UFACE(_i2z, ace_inner, pairpot, Eref)           
end
