
using ACEpotentials, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test 
using ACEbase: evaluate
using ACEbase.Testing: print_tf 

##

elements = [:Si,:O]

model = acemodel(; elements = elements, order = 3, totaldegree = 10)
pot = model.potential
mbpot = pot.components[2]

## 

ace1 = UltraFastACE.uface_from_ace1_inner(mbpot, 1; n_spl_points = 10_000)


## ------------------------------------

@info("Test Consistency of ACE1 with UFACE")
for ntest = 1:30 
   Nat = 12; r0 = 0.9 * rnn(:Si); r1 = 1.3 * rnn(:Si)
   Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
   iz0 = 1 
   z0 = JuLIP.Potentials.i2z(mbpot, 1)
   Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]

   v1 = ACEbase.evaluate(mbpot, Rs, Zs, z0)
   v2 = ACEbase.evaluate(ace1, Rs, Zs)

   print_tf(
      @test abs(v1 - v2) / (abs(v1) + abs(v2)) < 1e-10
            )  
end
println() 
