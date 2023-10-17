
using Interpolations, BenchmarkTools, StaticArrays

N = 32 
xspl = 0.0:0.01:3.0
spl = CubicSplineInterpolation(xspl, sin.(xspl))
f(m, x) = sin(m*x) * x * (3-x)

x = 1 + rand()
@btime ($spl)($x)

spl_tup = let N = N, spl_tup = ntuple(m -> CubicSplineInterpolation(xspl, f.(m, xspl)), N)
   x -> SVector(ntuple(m -> (spl_tup[m])(x), N))
end


##

xspl = 0.0:0.3:3.0
yspl = [ SVector{N}([f(m, xi) for m = 1:N]) for xi in xspl ]
spl_vec = CubicSplineInterpolation(xspl, yspl)
spl_vec(x)

##

@info("Stupid way")
@btime ($spl_tup)($x)
@info("SVector-spline way")
@btime ($spl_vec)($x)

