
using ACEpotentials, StaticArrays, Interpolations, BenchmarkTools, 
      LinearAlgebra, Polynomials4ML, SparseArrays, UltraFastACE
using ACEpotentials.ACE1: evaluate
P4ML = Polynomials4ML
C2R = UltraFastACE.ConvertC2R
import SpheriCart

##

zSi = AtomicNumber(:Si)
zO = AtomicNumber(:O)
# elements = [:Si,:O]
elements = [:Si,:O]
zelements = AtomicNumber.(elements)

model = acemodel(; elements = elements, order = 3, totaldegree = 10)
pot = model.potential
mbpot = pot.components[2]

## ------------------------------------

Nat = 12; r0 = 0.9 * rnn(:Si); r1 = 1.3 * rnn(:Si)
Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
rs = norm.(Rs)
iz0 = 1 
z0 = JuLIP.Potentials.i2z(mbpot, 1)
Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]
# Zs = fill(z0, Nat)


## --------------------
# transform the radial basis 
# this seems to be ok. 

b1p = mbpot.pibasis.basis1p
Rn_basis = mbpot.pibasis.basis1p.J


function make_radial_splines(Rn_basis, zlist; npoints = 100)
   @assert Rn_basis.envelope isa ACEpotentials.ACE1.OrthPolys.OneEnvelope
   rcut = Rn_basis.ru 
   rspl = range(0.0, rcut, length = npoints)
   function _make_rad_spl(z1, z0)
      yspl = [ SVector(evaluate(Rn_basis, r, z1, z0)...) for r in rspl ]
      return CubicSplineInterpolation(rspl, yspl)
   end
   spl = Dict([ (z1, z0) => _make_rad_spl(z1, z0) for z0 in zlist, z1 in zlist ])
   return spl  
end

spl = make_radial_splines(Rn_basis, zelements; npoints = 10_000)

for (z1, z0) in [(zSi, zSi), (zO, zO), (zSi, zO)]
   err = maximum( norm(evaluate(Rn_basis, r, z1, z0) - spl[(z1, z0)](r))
                     for r in rs )
   @show err
end
   
## --------------------

# spherical harmonics and the transformation 

L = b1p.SH.maxL
cYlm_basis_ace1 = b1p.SH 
rYlm_basis_sc = SpheriCart.SphericalHarmonics(L)
T_Ylm = C2R.r2c_transform(L) * C2R.sc2p4_transform(L)
D_T_Ylm = C2R._dict_Ylm_transform(T_Ylm)

𝐫 = @SVector randn(3)
Y_c = evaluate(cYlm_basis_ace1, 𝐫)
Y_r = SpheriCart.compute(rYlm_basis_sc, 𝐫)
Y_c ≈ T_Ylm * Y_r

## ------------------------------------
# we have three embeddings for the particles now 
# and can evaluate all these embeddings

zlist = JuLIP.Potentials.i2z.(Ref(mbpot), 1:length(elements))
Ez = reduce(vcat, [ (z .== zlist)' for z in Zs ])
Rn = reduce(vcat, [ spl[(Zs[j], z0)](norm(Rs[j]))' for j = 1:length(Rs) ])
Zlm = rYlm_basis_sc(Rs)

##  ------------------------------------

function make_ace1_AA_spec(mpot) 
   D_spec = mbpot.pibasis.inner[1].b2iAA
   spec_c = Vector{Vector{NamedTuple{(:z, :n, :l, :m), Tuple{AtomicNumber, Int, Int, Int}}}}(undef, length(D_spec))
   for (bb1, idx) in D_spec 
      bb = [ (z = b.z, n = b.n, l = b.l, m = b.m) for b in bb1.oneps ]
      spec_c[idx] = bb
   end  
   return spec_c  
end

function make_AA_transform(mbpot, D_T_Ylm)
   spec_AA_ace1 = make_ace1_AA_spec(mbpot)
   return C2R._AA_r2c_transform(spec_AA_ace1, D_T_Ylm; f = real)
end

AA_transform = make_AA_transform(mbpot, D_T_Ylm)

v1 = ACE1.evaluate(mbpot, Rs, Zs, z0)

lenAA1 = length(mbpot.coeffs[1])
AA_c = evaluate(mbpot.pibasis, Rs, Zs, z0)[1:lenAA1]
dot(real.(AA_c), real.(mbpot.coeffs[1])) ≈ v1


## ------------------------------------
# construct the real A basis 

AA_spec_r = AA_transform[:spec_r]
A_spec_r = sort(unique(reduce(vcat, AA_spec_r)))

# now we have to convert this into Ez, Rn, Zlm indices 
spec2i_Ez = Dict([zlist[i] => i for i = 1:length(zlist)]...)
spec2i_Rn = 1:size(Rn, 2)
spec2i_Ylm = Dict([ (l = P4ML.idx2lm(i)[1], m = P4ML.idx2lm(i)[2]) => i  
                   for i = 1:size(Zlm, 2) ]...)

inv_spec_A = Dict{NamedTuple, Int}() 
spec_A_inds = Vector{NTuple{3, Int}}(undef, length(A_spec_r))
for (i, b) in enumerate(A_spec_r)
   i_Ez = spec2i_Ez[b.z]
   i_Rn = spec2i_Rn[b.n]
   i_Ylm = spec2i_Ylm[(l = b.l, m = b.m)]
   inv_spec_A[b] = i 
   spec_A_inds[i] = (i_Ez, i_Rn, i_Ylm)
end

# generate the pooling layer 
A_basis = P4ML.PooledSparseProduct(spec_A_inds)
# and evaluate it 
A = A_basis((Ez, Rn, Zlm))

## ------------------------------------
# test this against the ACE1 A basis 
# this passes for single species but not multi-species. 

A_spec_c_ = ACE1.get_basis_spec(mbpot.pibasis.basis1p, iz0)
A_spec_c = [ (z = b.z, n = b.n, l = b.l, m = b.m) for b in A_spec_c_ ]
TA_r2c = C2R._A_transform(A_spec_c, A_spec_r, D_T_Ylm)
t_A_r = TA_r2c * A
A_c = ACE1.evaluate(mbpot.pibasis.basis1p, Rs, Zs, z0)

A_c ≈ t_A_r


## ------------------------------------
# construct the real AA basis 
# this means constructing kk = [k1, k2, ...] 
# with the ki pointing into an A vector. 

AA_spec_r = AA_transform[:spec_r]
inv_AA_spec_r = AA_transform[:inv_spec_r]
spec_AA_inds = Vector{Vector{Int}}(undef, length(AA_spec_r))
for (i, bb) in enumerate(AA_spec_r)
   spec_AA_inds[i] = sort([ inv_spec_A[b] for b in bb ])
end

AA_basis = P4ML.SparseSymmProd(spec_AA_inds)
AA_r = AA_basis(A)


AA_c 
AA_r_t = AA_transform[:T] * AA_r

real.(AA_c) ≈ AA_r_t

## transform the coefficients 

len_c = length(AA_c)
c_r = Dict(zO => AA_transform[:T]' * mbpot.coeffs[1], 
           zSi => AA_transform[:T]' * mbpot.coeffs[2])
dot(AA_r, c_r[zO]) ≈ v1           
