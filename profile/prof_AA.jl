using StaticArrays, BenchmarkTools, Polynomials4ML, ACEpotentials, UltraFastACE, 
      StaticPolynomials
using ACEbase.Testing: println_slim, print_tf
using Polynomials4ML: evaluate, evaluate!, release!
using LinearAlgebra: dot 
using ChainRules: rrule


##

pot = acemodel(; elements=[:Si,], order=3, totaldegree=12)
@show length(pot.basis)

basis = pot.basis.BB[2]
specm = basis.pibasis.inner[1].iAA2iA
ords = basis.pibasis.inner[1].orders
spect = [ tuple([specm[i, t] for t=1:ords[i]]...) for i = 1:length(ords) ]
nA = maximum(specm)
nAA = length(spect)
c = randn(nAA)
corr = SparseSymmProd(spect)
# corr = SparseSymmProdDAG()

AA_dot_p4ml = let corr=corr, c = c
   A -> ( AA = evaluate(corr, A); out = dot(AA, c); release!(AA); out )
end

AA_dot_poly = UltraFastACE.generate_AA_dot(spect, c)

##

A = @SVector randn(Float64, nA)
AA_dot_p4ml(A) ≈ AA_dot_poly(A)

@btime ($AA_dot_p4ml)($A)
@btime ($AA_dot_poly)($A)

##

@btime StaticPolynomials.gradient($AA_dot_poly, $A)

val, pb = rrule(evaluate, corr, A)
pb(c)[3]
@btime (∂A = ($pb)($c); release!(∂A));

# @btime Polynomials4ML.pullback_arg!($∂A, $∂AA, $corr, $AA) 

