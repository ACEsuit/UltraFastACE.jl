
using StaticArrays, BenchmarkTools, StrideArrays, Polynomials4ML

##


@generated function _update_A!(A, ::Val{SPEC}, R, Y) where {SPEC}
   code = Expr[]

   for i = 1:length(SPEC)
      n, k = SPEC[i]
      push!(code, Meta.parse("A[$i] = muladd(R[$n], Y[$k], A[$i])"))
   end
   return quote
      $(Expr(:block, code...))
   end
end

function static_A!(A, Nat, valspec, ::Val{N}, ::Val{K}) where {N, K}
   for i = 1:Nat 
      R = randn(SVector{N, Float64})
      Y = randn(SVector{K, Float64})
      _update_A!(A, valspec, R, Y)
   end
   return A
end

D = 12 
spec = filter(t -> t[1] + 2*sqrt(t[2]) <= D,  
             [ (n, k) for n = 1:D for k = 1:D^2 ])

N = maximum( b[1] for b in spec )
K = maximum( b[2] for b in spec )
Nat = 30 

valN = Val(N)
valK = Val(K)
valspec = Val(tuple(spec...))
A = @StrideArray zeros(length(spec))

static_A!(A, Nat, valspec, valN, valK)

@btime static_A!($A, $Nat, $valspec, $valN, $valK)



##

#
# in a new folder, start Julia 1.9 REPL, use ] to switch to 
# package manager, then run 
# ```
# activate .
# add BenchmarkTools, StrideArrays, Polynomials4ML, ChainRules 
# ```
# Then the following script should run. 
# To run it from terminal use 
# julia -O3 --project=. nameofscript.jl

using BenchmarkTools, StrideArrays, Polynomials4ML, ChainRules 

# specification of the pooled product 
#   [ (n1, k1), (n2, k2), ...]
# A[1] = ∑_j R[j, n1] * Y[j, k1]    etc ... 
Dtot = 12 
spec = filter(t -> t[1] + 2*sqrt(t[2]) <= Dtot,  
             [ (n, k) for n = 1:Dtot for k = 1:ceil(Int, Dtot^2/4+1) ])
pool = PooledSparseProduct(spec)

# length of R and Y bases, number of atoms in nhd 
N = maximum( b[1] for b in spec )
K = maximum( b[2] for b in spec )
Nat = 30 

# allocate some arrays for forward and backward pass 
R = PtrArray(randn(Nat, N))
Y = PtrArray(randn(Nat, K))
A = PtrArray(zeros(length(spec)))
∂R = PtrArray(randn(Nat, N))
∂Y = PtrArray(randn(Nat, K))
∂A = PtrArray(randn(length(spec)))

@info("Forward pass")
display(
      @benchmark evaluate!($A, $pool, $((R, Y)))
)

@info("Backward pass")
# this is normally hidden behind convenience layers 
display(@benchmark Polynomials4ML._pullback_evaluate!($((∂R, ∂Y)), $∂A, $pool, $((R, Y))))
