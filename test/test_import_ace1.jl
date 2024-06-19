
using ACE1, ACE1x, JuLIP, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test, ACEbase, Random 
using ACEbase: evaluate, evaluate_ed
using ACEbase.Testing: print_tf 

function rand_env(; Nat = rand(4:12), r0 = 0.8 * rnn(:Si), r1 = 2.0 * rnn(:Si))
      Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
      z0 = rand(AtomicNumber.(elements)) # JuLIP.Potentials.i2z(mbpot, 1)
      Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]
      return Rs, Zs, z0 
end

Random.seed!(1234)

##

elements = [:Si,:O]

# use totaldegree = 10, 12 for static aa 
#     totaldegree > 15 for dynamic aa

@info("Dynamic Model: totaldegree = 15")
model_dyn = acemodel(; elements = elements, 
                   order = 3, totaldegree = 15, 
                   Eref = Dict(:Si => -1.234, :O => -0.432))
pot_dyn = model_dyn.potential
pairpot_dyn = pot_dyn.components[1]
mbpot_dyn = pot_dyn.components[2]
pot1_dyn = pot_dyn.components[3]
@show length(mbpot_dyn.pibasis.inner[1])


@info("Static Model: totaldegree = 10")
model_st = acemodel(; elements = elements, 
                   order = 3, totaldegree = 10, 
                   Eref = Dict(:Si => -1.234, :O => -0.432))
pot_st = model_st.potential
pairpot_st = pot_st.components[1]
mbpot_st = pot_st.components[2]
pot1_st = pot_st.components[3]
@show length(mbpot_st.pibasis.inner[1])

## 

MODELS = [ (model_dyn, pot_dyn, pairpot_dyn, mbpot_dyn, pot1_dyn), 
           (model_st,  pot_st,  pairpot_st,  mbpot_st,  pot1_st) ]

##
# normalize the potential a bit so that all contributions are O(1) 
# pot1 will be O(1) by construction  

for (model, pot, pairpot, mbpot, pot1) in MODELS

Nsample = 10_000
pairpot.coeffs[:] = randn(length(pairpot.coeffs)) 
a = sum( evaluate(pairpot, rand_env()...) for _ = 1:Nsample ) / Nsample
pairpot.coeffs[:] /= a

c = randn(length(mbpot.pibasis))
mbpot_ = ACE1.PIPotential(mbpot.pibasis, c)
a = sum( evaluate(mbpot_, rand_env()...) for _ = 1:Nsample ) / Nsample
c[:] /= a 
mbpot_ = ACE1.PIPotential(mbpot.pibasis, c)
sum( evaluate(mbpot_, rand_env()...) for _ = 1:Nsample ) / Nsample

pot.components[2] = mbpot = mbpot_

# convert to UFACE format 
uf_ace = UltraFastACE.uface_from_ace1(pot; n_spl_points = 10_000)

# ------------------------------------

@info("Test Consistency of ACE1 with UFACE")

for ntest = 1:50 
   Rs, Zs, z0 = rand_env()

   v1_mb = evaluate(mbpot, Rs, Zs, z0)
   v1_pair = evaluate(pairpot, Rs, Zs, z0)
   v1_one = pot1.E0[chemical_symbol(z0)]
   v1 = v1_mb + v1_pair + v1_one 
   v2_pair = evaluate(uf_ace.pairpot, Rs, Zs, z0)
   v2 = evaluate(uf_ace, Rs, Zs, z0)
   v2_mb = v2 - v2_pair - uf_ace.E0s[z0]

   v1_pair ≈ v2_pair
   v1_mb ≈ v2_mb

   print_tf(
      @test abs(v1 - v2) / (abs(v1) + abs(v2)) < 1e-10
           )
end
println()


# check gradient ------------------------------

@info("test gradients")
for ntest = 1:30 
   Rs, Zs, z0 = rand_env()
   v1, dv1 = evaluate_ed(uf_ace, Rs, Zs, z0)
   U = randn(SVector{3, Float64}, length(dv1))

   F = t -> evaluate(uf_ace, Rs + t * U, Zs, z0)
   dF = t -> (dV = evaluate_ed(uf_ace, Rs + t * U, Zs, z0)[2]; 
            sum( dot(dv, u) for (dv, u) in zip(dV, U) ) )
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println()

end