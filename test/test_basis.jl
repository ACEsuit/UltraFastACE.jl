

using ACE1, ACE1x, JuLIP, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test, ACEbase
using ACEbase: evaluate, evaluate_ed
using ACEbase.Testing: print_tf 

function rand_env(; Nat = rand(4:12), r0 = 0.8 * rnn(:Si), r1 = 2.0 * rnn(:Si))
      Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
      z0 = rand(AtomicNumber.(elements)) # JuLIP.Potentials.i2z(mbpot, 1)
      Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]
      return Rs, Zs, z0 
end

eval_stack(ace1, Rs, Zs, z0) = (   
            evaluate(ace1.components[1], Rs, Zs, z0)
          + evaluate(ace1.components[2], Rs, Zs, z0)
          + ace1.components[3].E0[chemical_symbol(z0)] )

##

elements = [:Si,:O]

model = acemodel(; elements = elements, 
                   order = 3, totaldegree = 10, 
                   Eref = Dict(:Si => -1.234, :O => -0.432))
ace1 = model.potential                   
uf_ace = UltraFastACE.uface_from_ace1(ace1; n_spl_points = 10_000)

## ------------------------------------

Rs, Zs, z0 = rand_env()
v1 = eval_stack(ace1, Rs, Zs, z0)
v2 = evaluate(uf_ace, Rs, Zs, z0)
@show v1 ≈ v2

## produce a basis 


module UFB

using ACE1, ACE1x, JuLIP, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test, ACEbase,
      Polynomials4ML, ObjectPools

import ACEbase: evaluate       

struct LinearACEInnerBasis{TR, TY, TA, TAA}
   rbasis::TR
   ybasis::TY
   abasis::TA
   aabasis::TAA
   # ---------- admin and meta-data 
   meta::Dict{String, Any}
   pool::TSafe{ArrayPool{FlexArrayCache}}
end

LinearACEInnerBasis(rbasis, ybasis, abasis, aabasis, 
                     meta = Dict{String, Any}()) = 
      LinearACEInnerBasis(rbasis, ybasis, abasis, aabasis, meta, 
                           TSafe(ArrayPool(FlexArrayCache)))

Base.length(ufb_in::LinearACEInnerBasis) = length(ufb_in.aabasis)

struct LinearACEBasis{NZ, INNER}
   _i2z::NTuple{NZ, Int}
   ace_inner::INNER
   # pairbasis::PAIR
   # E0s::Dict{Int, Float64}
   # ---------- 
   meta::Dict{String, Any}
   pool::TSafe{ArrayPool{FlexArrayCache}}
end

LinearACEBasis(_i2z, ace_inner, meta = Dict{String, Any}()) = 
         LinearACEBasis(_i2z, ace_inner, meta, 
                       TSafe(ArrayPool(FlexArrayCache)))

Base.length(basis::LinearACEBasis) = sum(length, basis.ace_inner) 

function inner_basis_from_ufpot(ufpot_in)
   AA_spec = ufpot_in.meta["AA_spec"]
   ufb_in = UFB.LinearACEInnerBasis(
               ufpot_in.rbasis, 
               ufpot_in.ybasis, 
               ufpot_in.abasis, 
               SparseSymmProd(AA_spec), 
               ufpot_in.meta)
   return ufb_in
end

function basis_from_ufpot(ufpot)
   NZ = length(ufpot.ace_inner)
   t_inner = ntuple(iz -> inner_basis_from_ufpot(ufpot.ace_inner[iz]), NZ)
   coeffs = ntuple(iz -> ufpot.ace_inner[iz].meta["AA_coeffs"], NZ)
   meta = Dict{String, Any}(ufpot.meta...)
   basis = LinearACEBasis(ufpot._i2z, t_inner, meta)
   ps = (coeffs_inner = coeffs, )
   return basis, ps 
end

function evaluate_basis(ufb_in::LinearACEInnerBasis, Rs, Zs)
   TF = eltype(eltype(Rs))
   rbasis = ufb_in.rbasis 
   NZ = length(rbasis._i2z)
   
   # embeddings    
   Ez = UltraFastACE.embed_z(ufb_in, Rs, Zs)
   Rn = UltraFastACE.evaluate(ufb_in, rbasis, Rs, Zs)
   Zlm = UltraFastACE.evaluate_ylm(ufb_in, Rs)
   # pooling 
   A = ufb_in.abasis((unwrap(Ez), unwrap(Rn), unwrap(Zlm)))
   # n correlations
   AA = ufb_in.aabasis(A)

   # release the borrowed arrays 
   release!(Zlm)
   release!(Rn)
   release!(Ez)
   release!(A)

   return AA
end

function evaluate(basis::LinearACEBasis, Rs, Zs, z0, ps::NamedTuple)
   iz0 = UltraFastACE._z2i(basis, z0)
   AA = evaluate_basis(basis.ace_inner[iz0], Rs, Zs)
   return dot(AA, ps.coeffs_inner[iz0])
end

function _get_range(basis::LinearACEBasis, z0)
   iz0 = UltraFastACE._z2i(basis, z0) 
   lens_inner = length.(basis.ace_inner)
   start = sum(lens_inner[1:iz0-1]) + 1
   stop = sum(lens_inner[1:iz0])
   return start:stop
end

function evaluate_basis(basis::LinearACEBasis, Rs, Zs, z0)
   iz0 = UltraFastACE._z2i(basis, z0)
   AA_iz0 = evaluate_basis(basis.ace_inner[iz0], Rs, Zs)
   AA_all = zeros(eltype(AA_iz0), length(basis))
   AA_all[_get_range(basis, z0)] = AA_iz0
   release!(AA_iz0) 
   return AA_all 
end


function evaluate_d_basis(ufb_in::LinearACEInnerBasis, Rs, Zs)
   TF = eltype(eltype(Rs))
   TV = eltype(Rs) 
   rbasis = ufb_in.rbasis 
   NZ = length(rbasis._i2z)
   nX = length(Rs) 
   
   # embeddings    
   Ez = UltraFastACE.embed_z(ufb_in, Rs, Zs)
   Rn, ∂Rn = UltraFastACE.evaluate_ed(ufb_in, rbasis, Rs, Zs)
   Zlm, ∂Zlm = UltraFastACE.evaluate_ylm_ed(ufb_in, Rs)

   # because the pfwd implementation doesn't know about categorial 
   # variables we need to create a zero-valued array for dEz 
   ∂Ez = zeros(TV, size(Ez))

   # pooling 
   A, ∂A = Polynomials4ML.pfwd_evaluate(ufb_in.abasis, 
                           (unwrap(Ez),  unwrap(Rn),  unwrap(Zlm)), 
                           (unwrap(∂Ez), unwrap(∂Rn), unwrap(∂Zlm)))

   # n correlations
   AA, ∂AA = Polynomials4ML.pfwd_evaluate(ufb_in.aabasis, A, ∂A) 

   # release the borrowed arrays 
   release!(Zlm); release!(∂Zlm)
   release!(Rn); release!(∂Rn)
   release!(Ez); release!(∂Ez)
   release!(A); release!(∂A) 

   return AA, ∂AA
end


function evaluate_d_basis(basis::LinearACEBasis, Rs, Zs, z0)
   iz0 = UltraFastACE._z2i(basis, z0)
   AA_iz0, ∂AA_iz0 = evaluate_d_basis(basis.ace_inner[iz0], Rs, Zs)
   AA_all = zeros(eltype(AA_iz0), length(basis))
   ∂AA_all = zeros(eltype(∂AA_iz0), length(basis), size(∂AA_iz0, 2))
   rg = _get_range(basis, z0)
   AA_all[rg] = AA_iz0
   ∂AA_all[rg, :] = ∂AA_iz0
   release!(AA_iz0)
   release!(∂AA_iz0) 
   return AA_all, ∂AA_all
end


end

##

Rs, Zs, z0 = rand_env()
iz0 = UltraFastACE._z2i(uf_ace, z0)
ufpot_in1 = uf_ace.ace_inner[iz0]
ufb_in1 = UFB.inner_basis_from_ufpot(ufpot_in1)
coeffs = ufpot_in1.meta["AA_coeffs"]

v1 = ACEbase.evaluate(ufpot_in1, Rs, Zs)
AA = UFB.evaluate_basis(ufb_in1, Rs, Zs)
v2 = dot(coeffs, AA)
v1 ≈ v2

AA2, ∂AA2 = UFB.evaluate_d_basis(ufb_in1, Rs, Zs)

using ForwardDiff
using LinearAlgebra: I 

Us = randn(SVector{3, Float64}, length(Rs))
F = t -> UFB.evaluate_basis(ufb_in1, Rs + t * Us, Zs)
adF = ForwardDiff.derivative(F, 0.0)
dF = [ sum(dot(∂AA2[i, j], Us[j]) for j = 1:length(Rs)) for i = 1:length(AA) ]
@show adF ≈ dF

## 

basis, ps = UFB.basis_from_ufpot(uf_ace)
coeffs = vcat(ps.coeffs_inner...)
AA_all = UFB.evaluate_basis(basis, Rs, Zs, z0)
dot(coeffs, AA_all) ≈ v2

AA_all2, ∂AA_all2 = UFB.evaluate_d_basis(basis, Rs, Zs, z0)
AA_all2 ≈ AA_all

Us = randn(SVector{3, Float64}, length(Rs))
F = t -> UFB.evaluate_basis(basis, Rs + t * Us, Zs, z0)
adF = ForwardDiff.derivative(F, 0.0)
dF = [ sum(dot(∂AA_all2[i, j], Us[j]) for j = 1:length(Rs)) 
        for i = 1:length(AA_all2) ]
@show adF ≈ dF

##

