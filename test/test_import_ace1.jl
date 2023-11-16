
using ACEpotentials, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test 
using ACEbase: evaluate
using ACEbase.Testing: print_tf 

##

elements = [:Si,:O]

model = acemodel(; elements = elements, 
                   order = 3, totaldegree = 10, 
                   E0 = Dict(:Si => -1.234, :O => -0.432))
pot = model.potential
pairpot = pot.components[1]
mbpot = pot.components[2]

# conver to UFACE format 

# UltraFastACE.make_pairpot_splines(pairpot)
uf_ace = UltraFastACE.uface_from_ace1(pot; n_spl_points = 10_000)

## ------------------------------------

@info("Test Consistency of ACE1 with UFACE")
for ntest = 1:50 
   Nat = rand(4:12); r0 = 0.8 * rnn(:Si); r1 = 2.0 * rnn(:Si)
   Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
   z0 = rand(AtomicNumber.(elements)) # JuLIP.Potentials.i2z(mbpot, 1)
   Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]

   v1_mb = evaluate(mbpot, Rs, Zs, z0)
   v1_pair = evaluate(pairpot, Rs, Zs, z0)
   v1 = v1_mb + v1_pair 
   v2_pair = evaluate(uf_ace.pairpot, Rs, Zs, z0)
   v2_mb = v2 - v2_pair
   v2 = evaluate(uf_ace, Rs, Zs, z0)

   v1_mb ≈ v2_mb

   print_tf(
      @test abs(v1 - v2) / (abs(v1) + abs(v2)) < 1e-10
            )  
end
println() 

