using ACEpotentials, StaticArrays, Interpolations, BenchmarkTools, 
      LinearAlgebra, Polynomials4ML, SparseArrays, UltraFastACE
using ACEpotentials.ACE1: evaluate
P4ML = Polynomials4ML
C2R = UltraFastACE.ConvertC2R
import SpheriCart

##

L = 3
# cYlm_basis_ace1 = b1p.SH 
cYlm_basis_p4 = CYlmBasis(L)
rYlm_basis_p4 = RYlmBasis(L)
rYlm_basis_sc = SpheriCart.SphericalHarmonics(L)

𝐫 = @SVector randn(3)
# Y1 = evaluate(cYlm_basis_ace1, 𝐫)
Y2 = evaluate(cYlm_basis_p4, 𝐫)
Y3 = evaluate(rYlm_basis_p4, 𝐫)
Y4 = SpheriCart.compute(rYlm_basis_sc, 𝐫)

T_r2c = C2R.r2c_transform(L)
T_r2c * Y3 ≈ Y2

T_sc2p4 = C2R.sc2p4_transform(L)
T_sc2p4 * Y4 ≈ Y3

# this is the actual transform we want to convert from ACE1 to SpheriCart
T = T_r2c * T_sc2p4
T * Y4 ≈ Y2

##
# testing the transform of products of Ylms 

D_T = C2R._dict_Ylm_transform(T)
tt, cache = C2R._prodYlm_transform((1, 2, 2), (-1, 1, 2), D_T)


function rand_tuple(L)
   N = rand(2:5)
   ll = rand(0:L, N)
   mm = [rand(-ll[t]:ll[t]) for t = 1:N]
   while sum(mm) != 0 
      if sum(mm) > 0 
         i = findmax(mm)[2] 
         mm[i] -= 1 
      else
         i = findmin(mm)[2]
         mm[i] += 1
      end
   end
   return tuple(ll...), tuple(mm...)
end

eval_prod(ll, mm, Y) = prod(
   Y[P4ML.lm2idx(l, m)] for (l, m) in zip(ll, mm) )
   

L = 5 
cYlm_basis_p4 = CYlmBasis(L)
rYlm_basis_sc = SpheriCart.SphericalHarmonics(L)
𝐫 = rand(SVector{3, Float64})
ll, mm = rand_tuple(L)
Y_p4 = evaluate(cYlm_basis_p4, 𝐫)
Y_sc = SpheriCart.compute(rYlm_basis_sc, 𝐫)

aa_c = eval_prod(ll, mm, Y_p4)

T1 = C2R.r2c_transform(L) * C2R.sc2p4_transform(L)
D_T1 = C2R._dict_Ylm_transform(T1)
TT_llmm = C2R._prodYlm_transform(ll, mm, D_T1)[1]
aa_r = sum(t[2] * eval_prod(ll, t[1], Y_sc) for t in TT_llmm)

TTr_llmm = C2R._prodYlm_transform(ll, mm, D_T1; f = real)[1]

aa_rr = sum(real(t[2]) * eval_prod(ll, t[1], Y_sc) for t in TT_llmm)

aa_c ≈ aa_r
real(aa_c) ≈ aa_rr

##
# now try to transform an actual ACE1 model 

zSi = AtomicNumber(:Si)
zO = AtomicNumber(:O)
elements = [:Si,:O]

model = acemodel(; elements = elements, order = 3, totaldegree = 10)
pot = model.potential
mbpot = pot.components[2]

D_spec = mbpot.pibasis.inner[1].b2iAA
spec_c = Vector{Vector{NamedTuple{(:z, :n, :l, :m), Tuple{AtomicNumber, Int, Int, Int}}}}(undef, length(D_spec))
for (bb1, idx) in D_spec 
   bb = [ (z = b.z, n = b.n, l = b.l, m = b.m) for b in bb1.oneps ]
   spec_c[idx] = bb
end

AA_transform = C2R._AA_r2c_transform(spec_c, D_T1; f = real)

length(AA_transform[:spec_c])
length(AA_transform[:spec_r])
size(AA_transform[:T])
AA_transform[:T].nzval

## 

