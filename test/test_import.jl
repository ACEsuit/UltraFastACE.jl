
using ACEpotentials, StaticArrays, Interpolations, BenchmarkTools, 
      LinearAlgebra, Polynomials4ML, SparseArrays
using ACEpotentials.ACE1: evaluate
P4ML = Polynomials4ML
import SpheriCart

zSi = AtomicNumber(:Si)
zO = AtomicNumber(:O)
elements = [:Si,:O]

model = acemodel(; elements = elements, order = 3, totaldegree = 10)
pot = model.potential
mbpot = pot.components[2]

b1p = mbpot.pibasis.basis1p
Rn_basis = mbpot.pibasis.basis1p.J


function make_radial_splines(Rn_basis; npoints = 100)
   @assert Rn_basis.envelope isa ACEpotentials.ACE1.OrthPolys.OneEnvelope
   rcut = Rn_basis.ru 
   rspl = range(0.0, rcut, length = npoints)
   yspl = [ SVector(evaluate(Rn_basis, r, zSi, zSi)...) for r in rspl ]
   spl = CubicSplineInterpolation(rspl, yspl)
end

spl = make_radial_splines(Rn_basis)

norm(evaluate(Rn_basis, 1.0, zSi, zSi) - spl(1.0))

# the above seems to work, we can return to it once the b1p import is sorted...

## --------------------

# spherical harmonics and the transformation 

L = b1p.SH.maxL
cYlm_basis_ace1 = b1p.SH 
cYlm_basis_p4 = CYlmBasis(L)
rYlm_basis_p4 = RYlmBasis(L)
rYlm_basis_sc = SpheriCart.SphericalHarmonics(L)

𝐫 = @SVector randn(3)
Y1 = evaluate(cYlm_basis_ace1, 𝐫)
Y2 = evaluate(cYlm_basis_p4, 𝐫)
Y3 = evaluate(rYlm_basis_p4, 𝐫)
Y4 = SpheriCart.compute(rYlm_basis_sc, 𝐫)

function r2c_transform(L)
   T = zeros(ComplexF64, (L+1)^2, (L+1)^2)
   for l = 0:L 
      # m = 0
      i_l0 = P4ML.lm2idx(l, 0)
      T[i_l0, i_l0] = 1.0 
      for m = 1:l
         i_lm  = P4ML.lm2idx(l,  m)
         i_l⁻m = P4ML.lm2idx(l, -m)
         T[ i_lm,  i_lm] = 1/sqrt(2) 
         T[ i_lm, i_l⁻m] = -im/sqrt(2) 
         T[i_l⁻m,  i_lm] = (-1)^m/sqrt(2) 
         T[i_l⁻m, i_l⁻m] = (-1)^m*im/sqrt(2) 
      end
   end
   return sparse(T)
end

function sc2p4_transform(L) 
   D = zeros((L+1)^2)
   for l = 0:L
      for m = -l:-1
         i_lm = P4ML.lm2idx(l, m)
         D[i_lm] = (-1)^(m+1)
      end
      i_l0 = P4ML.lm2idx(l, 0)
      D[i_l0] = 1
      for m = 1:l
         i_lm = P4ML.lm2idx(l, m)
         D[i_lm] = (-1)^(m)
      end
   end
   return Diagonal(D)
end

Y1 ≈ Y2

T_r2c = r2c_transform(L)
T_r2c * Y3 ≈ Y2

T_sc2p4 = sc2p4_transform(L)
T_sc2p4 * Y4 ≈ Y3

T = T_r2c * T_sc2p4
T * Y4 ≈ Y1
Tinv = sparse(round.(pinv(Matrix(T)); digits=15))

##  ------------------------------------

aa_basis = mbpot.pibasis
aa_spec_1 = ACE1.get_basis_spec(aa_basis, 1)
aa_llmm = [ [(b.l, b.m) for b in bb.oneps] for bb in aa_spec_1 ]

inv_spec = Dict{Any, Int}() 
for i = 1:length(aa_spec_1)
   inv_spec[aa_spec_1[i]] = i
end


# Y_ll^mm 
#   = ∏_t Y_lt^mt
#   = ∏_t  ∑_kt T[(lt, mt), (lt, kt)] * Z_lt^kt 
#   = ∑_{k1,k2,...} { ∏_t T[(lt, mt), (lt, kt)] } ∏_t Z_lt^kt
#   = ∑_{k1,k2,...} S^ll_{mm, kk}  Z_ll^kk
# this will give us the transformation from AA_r -> AA_c 

lenAA = length(aa_spec_1)
S = zeros(ComplexF64, lenAA, lenAA)
for i = 50:60
   bb = aa_spec_1[i] 
   ll = [ b.l for b in bb.oneps ]
   mm = [ b.m for b in bb.oneps ]
   N = length(ll)

   transforms = [] 
   for t = 1:N 
      lt = ll[t]
      mt = mm[t]
      i_ltmt = P4ML.lm2idx(lt, mt)
      push!(transforms, findall(!iszero, T[i_ltmt, :]))
   end

   for rows_N in Iterators.product(transforms...)
      
      # c = prod([ c for (_, c) in rows_N ])
      # get back the global indices
      # S[i, rows] = c
      kk = [ P4ML.idx2lm(it)[2] for it in rows_N ]
      @show kk 
   end
end

## -------------------------------------

Nat = 10 
Rs = [ (@SVector randn(3)) for _ = 1:Nat ]
Zs = [ rand([zSi, zO]) for _ = 1:Nat ]
z0 = zSi 

evaluate(b1p, Rs, Zs, z0)

spec1p = ACE1.get_basis_spec(b1p, zSi)



