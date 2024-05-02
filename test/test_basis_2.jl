

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
   ps = (coeffs_inner = coeffs, )
   return LinearACEBasis(ufpot._i2z, t_inner, meta), 
          ps 
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

using ReverseDiff, Zygote

function evaluate(ufbas::LinearACEBasis, Rs, Zs, z0, ps::NamedTuple)
   iz0, AA = Zygote.ignore() do 
      iz0 = UltraFastACE._z2i(ufbas, z0)
      ( iz0, evaluate_basis(ufbas.ace_inner[iz0], Rs, Zs))
   end
   return dot(AA, ps.coeffs_inner[iz0])
end


# function evaluate_ed(ace::UFACE, Rs, Zs, z0)
#    ∇φ = acquire!(ace.pool, :out_dEs, (length(Rs),), eltype(Rs))
#    return evaluate_ed!(∇φ, ace, Rs, Zs, z0)
# end

# function evaluate_ed!(∇φ, ace::UFACE, Rs, Zs, z0)
#    i_z0 = _z2i(ace, z0)
#    ace_inner = ace.ace_inner[i_z0]
#    φ, _ = evaluate_ed!(∇φ, ace_inner, Rs, Zs)
#    add_evaluate_d!(∇φ, ace.pairpot, Rs, Zs, z0)
#    φ += ace.E0s[z0] + evaluate(ace.pairpot, Rs, Zs, z0)
#    return φ, ∇φ
# end


# function ACEbase.evaluate_ed!(∇φ, ace::UFACE_inner, Rs, Zs)
#    TF = eltype(eltype(Rs))
#    rbasis = ace.rbasis 
#    NZ = length(rbasis._i2z)

#    # embeddings    
#    # element embedding  (there is no gradient)
#    Ez = embed_z(ace, Rs, Zs)

#    # radial embedding 
#    Rn, dRn = evaluate_ed(ace, rbasis, Rs, Zs)

#    # angular embedding 
#    Zlm, dZlm = evaluate_ylm_ed(ace, Rs)

#    # pooling 
#    A = ace.abasis((Ez, Rn, Zlm))

#    # n correlations - compute with gradient, do it in-place 
#    ∂φ_∂A = acquire!(ace.pool, :∂A, size(A), TF)
#    φ = evaluate_and_gradient!(∂φ_∂A, ace.aadot, A)
   
#    # backprop through A  =>  this part could be done more nicely I think
#    ∂φ_∂Ez = BlackHole(TF) 
#    # ∂φ_∂Ez = zeros(TF, size(Ez))
#    ∂φ_∂Rn = acquire!(ace.pool, :∂Rn, size(Rn), TF)
#    ∂φ_∂Zlm = acquire!(ace.pool, :∂Zlm, size(Zlm), TF)
#    fill!(∂φ_∂Rn, zero(TF))
#    fill!(∂φ_∂Zlm, zero(TF))
#    P4ML._pullback_evaluate!((∂φ_∂Ez, unwrap(∂φ_∂Rn), unwrap(∂φ_∂Zlm)), 
#                              unwrap(∂φ_∂A), 
#                              ace.abasis, 
#                              (unwrap(Ez), unwrap(Rn), unwrap(Zlm)); 
#                              sizecheck=false)

#    # backprop through the embeddings 
#    # depending on whether there is a bottleneck here, this can be 
#    # potentially implemented more efficiently without needing writing/reading 
#    # (to be investigated where the bottleneck is)
   
#    # we just ignore Ez (hence the black hole)

#    # backprop through Rn 
#    # We already computed the gradients in the forward pass
#    fill!(∇φ, zero(SVector{3, TF}))
#    @inbounds for n = 1:size(Rn, 2)
#       @simd ivdep for j = 1:length(Rs)
#          ∇φ[j] += ∂φ_∂Rn[j, n] * dRn[j, n]
#       end
#    end

#    # ... and Ylm 
#    @inbounds for i_lm = 1:size(Zlm, 2)
#       @simd ivdep for j = 1:length(Rs)
#          ∇φ[j] += ∂φ_∂Zlm[j, i_lm] * dZlm[j, i_lm]
#       end
#    end

#    # release the borrowed arrays 
#    release!(Zlm); release!(dZlm)
#    release!(Rn); release!(dRn)
#    release!(Ez)
#    release!(A)
#    release!(∂φ_∂Rn); release!(∂φ_∂Zlm); release!(∂φ_∂A)

#    return φ, ∇φ 
# end


end

##

Rs, Zs, z0 = rand_env()
iz0 = UltraFastACE._z2i(uf_ace, z0)
ufpot_in1 = uf_ace.ace_inner[iz0]
ufb_in1 = UFB.inner_basis_from_ufpot(ufpot_in1)
coeffs = ufpot_in1.meta["AA_coeffs"]
ufbas, ps = UFB.basis_from_ufpot(uf_ace)

v1 = ACEbase.evaluate(ufpot_in1, Rs, Zs)
AA = UFB.evaluate_basis(ufb_in1, Rs, Zs)
v2 = dot(coeffs, AA)
v3 = evaluate(ufbas, Rs, Zs, z0, ps)

@show v1 ≈ v2 ≈ v3

## 

using Zygote, ReverseDiff, Optimisers, ForwardDiff

pvec, _rest = destructure(ps)
@time AA_fd = ForwardDiff.gradient(_p -> evaluate(ufbas, Rs, Zs, z0, _rest(_p)), pvec)
@time AA_rd = ReverseDiff.gradient(_p -> evaluate(ufbas, Rs, Zs, z0, _rest(_p)), pvec)
@time AA_zy = Zygote.gradient(_ps -> evaluate(ufbas, Rs, Zs, z0, _ps), ps)
@time AA1 = UFB.evaluate_basis(ufbas.ace_inner[1], Rs, Zs)
AA2 = UFB.evaluate_basis(ufbas.ace_inner[2], Rs, Zs)
AA_man = [ (iz0 == 1) * AA1; (iz0 == 2) * AA2 ]
AA_ad == AA_man

# unfortunately the performance of both reversediff and zygote are 
# really aweful here. This is very sad.

## 
