using StaticArrays, BenchmarkTools, Zygote 

function eval_AA_dot(A, spec, c)
   T = promote_type(eltype(A), eltype(c))
   out = zero(T)
   @inbounds for (i, (k1, k2)) in enumerate(spec)
      out += A[k1] * A[k2] * c[i] 
   end
   return out 
end

@generated function uf1_eval_AA_dot(A::SVector{N, T}, ::Val{spec}, ::Val{c}) where {N, T, spec, c}
   code = Expr[] 
   for (ci, (k1, k2)) in zip(c, spec)
      push!(code, :(out += A[$k1] * A[$k2] * $ci))
   end
   quote
      out = zero($T)
      $(Expr(:block, code...))
      return out
   end
end

##

nA = 100
nAA = 1000
spec = [ (rand(1:nA), rand(1:nA)) for _ = 1:nAA ]
tspec = tuple(spec...)

A = randn(SVector{nA, Float64})
c = randn(SVector{nAA, Float64})

A1 = Vector(A)
c1 = Vector(c)

tc = tuple(c...)

@btime eval_AA_dot($A, $spec, $c)
@btime eval_AA_dot($A1, $spec, $c1)
@btime eval_AA_dot($A, $tspec, $tc)
@btime uf1_eval_AA_dot($A, $(Val(tspec)), $(Val(tc)))


f1 = let spec=tspec, c = tc 
   A -> eval_AA_dot(A, spec, c)
end

f2 = let spec=Val(tspec), c = Val(tc) 
   A -> uf1_eval_AA_dot(A, spec, c)
end

##

# Zygote.gradient(f1, A)
# Zygote.gradient(f2, A)  # never finishes!!! try Enzme? 
# @time Zygote.gradient(f1, A)
# @time Zygote.gradient(f2, A)

##

using DynamicPolynomials: @polyvar 
using StaticPolynomials

@polyvar Ap[1:100]
p_AA_dot = Polynomial(f1(Ap))

@btime p_AA_dot($A)
StaticPolynomials.gradient(p_AA_dot, A)
@btime StaticPolynomials.gradient($p_AA_dot, $A)