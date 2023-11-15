
using ACEpotentials, StaticArrays, BenchmarkTools, 
      LinearAlgebra, UltraFastACE
using ACEbase: evaluate, evaluate!, evaluate_d, evaluate_d!, 
               evaluate_ed, evaluate_ed!

##

elements = [:Si,:O]

model = acemodel(; elements = elements, order = 3, totaldegree = 12)
pot = model.potential
mbpot = pot.components[2]

# convert to UFACE format 
uf_ace = UltraFastACE.uface_from_ace1(mbpot; n_spl_points = 100)

## ------------------------------------


Nat = 30; 
r0 = 0.9 * rnn(:Si); r1 = 1.3 * rnn(:Si)
Rs = [ (r0 + (r1 - r0) * rand()) * ACE1.Random.rand_sphere() for _=1:Nat ]
z0 = rand(AtomicNumber.(elements)) 
Zs = [ rand(AtomicNumber.(elements)) for _ = 1:Nat ]

v1 = evaluate(mbpot, Rs, Zs, z0)
v2 = evaluate(uf_ace, Rs, Zs, z0)

dv1 = evaluate_d(mbpot, Rs, Zs, z0)
dv2 = deepcopy(dv1) 
v3, _ = evaluate_ed!(dv2, uf_ace, Rs, Zs, z0)

@show abs(v3 - v2)
@show norm(dv1-dv2) / norm(dv1)

##

@info("ACE1 - allocating")
@btime evaluate($mbpot, $Rs, $Zs, $z0)

@info("ACE1 - pre-allocated")
tmp = ACE1.alloc_temp(mbpot, length(Rs))
@btime evaluate!($tmp, $mbpot, $Rs, $Zs, $z0)

@info("UF_ACE")
@btime evaluate($uf_ace, $Rs, $Zs, $z0)

## 

# @profview let uf_ace = uf_ace, Rs = Rs, Zs = Zs, z0 = z0
#    for iter = 1:400_000
#       evaluate(uf_ace, Rs, Zs, z0)
#    end
# end

##
# profiling the gradient code 

@info("profiling the gradient code")

@info("ACE1 - allocating")
@btime evaluate_d($mbpot, $Rs, $Zs, $z0)

@info("ACE1 - preallocated arrays")
tmpd = ACE1.alloc_temp_d(mbpot, length(Rs))
dEs1 = zeros(SVector{3, Float64}, length(Rs))
@btime evaluate_d!($dEs1, $tmpd, $mbpot, $Rs, $Zs, $z0)

@info("UF_ACE")
dEs2 = zeros(SVector{3, Float64}, length(Rs))
evaluate_ed!(dEs2, uf_ace, Rs, Zs, z0)
@btime evaluate_ed!($dEs2, $uf_ace, $Rs, $Zs, $z0)

##

@profview let uf_ace = uf_ace, Rs = Rs, Zs = Zs, z0 = z0, dEs2 = dEs2 
   for iter = 1:2_000
      evaluate_ed!(dEs2, uf_ace, Rs, Zs, z0)
   end
end