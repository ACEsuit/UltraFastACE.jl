
using ACE1, ACE1x, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE, Test, ACEbase
using ACEbase: evaluate, evaluate_ed
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
uf_ace = UltraFastACE.uface_from_ace1(pot; n_spl_points = 100)

##

function rand_struct(; rep = rand(1:3), cubic=true, rattle = 0.1)
   zz = [AtomicNumber(:Si), AtomicNumber(:O)]
   at = bulk(:Si, cubic=cubic) * rep
   at.Z[:] = rand(zz, length(at))
   rattle!(at, rattle)
   return at 
end

at = rand_struct()

jpot = UltraFastACE.UFACE_JuLIP(uf_ace)
cutoff(jpot)

JuLIP.usethreads!(false)

##
# energy(pot, at) ≈ 
energy(jpot, at) ≈ UltraFastACE.energy_new(uf_ace, at)
@btime energy($pot, $at)
@btime energy($jpot, $at)
@btime UltraFastACE.energy_new($uf_ace, $at)

##

F1 = forces(pot, at)
F2 = forces(jpot, at)
@show norm(F1 - F2)

@info("ACE1 Calculator")
# display(@benchmark forces($pot, $at))
@btime forces($pot, $at)
@info("UF_ACE Calculator")
# display(@benchmark forces($jpot, $at))
@btime forces($jpot, $at)
@info("New calculator")
F = zeros(SVector{3, Float64}, length(at))
@btime UltraFastACE.forces_new!($F, $uf_ace, $at)
@btime UltraFastACE.forces_new($uf_ace, $at)

## 

JuLIP.usethreads!(true)
at = rand_struct(; rep = 12)

##

@info("neighbourlist")
@time JuLIP.neighbourlist(at, 6.0)
@info("Forces - Multithreaded - JuLIP")
display(@benchmark forces(jpot, at))
@info("Forces - Single threaded - new")
display(@benchmark UltraFastACE.forces_new(uf_ace, at))
@info("Forces - Multithreaded - new")
display(@benchmark UltraFastACE.forces_new_mt(uf_ace, at))

# @time forces(pot, at)
# @time forces(jpot, at)

##

@profview let uf_ace = uf_ace, at = at, F = F 
   for ntest = 1:4_000
      # UltraFastACE.energy_new(uf_ace, at)
      UltraFastACE.forces_new!(F, uf_ace, at)
   end
end

