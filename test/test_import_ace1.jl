
using ACEpotentials, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test 
using ACEbase: evaluate
using ACEbase.Testing: print_tf 

function rand_env(; Nat = rand(4:12), r0 = 0.8 * rnn(:Si), r1 = 2.0 * rnn(:Si))
      Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
      z0 = rand(AtomicNumber.(elements)) # JuLIP.Potentials.i2z(mbpot, 1)
      Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]
      return Rs, Zs, z0 
end

##

elements = [:Si,:O]

model = acemodel(; elements = elements, 
                   order = 3, totaldegree = 10, 
                   Eref = Dict(:Si => -1.234, :O => -0.432))
pot = model.potential
pairpot = pot.components[1]
mbpot = pot.components[2]
pot1 = pot.components[3]

##
# normalize the potential a bit so that all contributions are O(1) 
# pot1 will be O(1) by construction  

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

## ------------------------------------

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
