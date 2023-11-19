
module UFtesting

function rand_env(; Nat = rand(4:12), r0 = 0.8 * rnn(:Si), r1 = 2.0 * rnn(:Si))
   Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
   z0 = rand(AtomicNumber.(elements)) # JuLIP.Potentials.i2z(mbpot, 1)
   Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]
   return Rs, Zs, z0 
end

function normalize_potential()
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



end