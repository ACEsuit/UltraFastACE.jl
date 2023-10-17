
using Interpolations, BenchmarkTools, StaticArrays

xspl = 0.0:0.01:3.0
spl = CubicSplineInterpolation(xspl, sin.(xspl))
f(m, x) = sin(m*x) * x * (3-x)

x = 1 + rand()
@btime ($spl)($x)

spl_tup = let spl_tup = ntuple(m -> CubicSplineInterpolation(xspl, f.(m, xspl)), 10)
   x -> SVector(ntuple(m -> (spl_tup[m])(x), 10))
end


##

xspl = 0.0:0.3:3.0
yspl = [ SVector{10}([f(m, xi) for m = 1:10]) for xi in xspl ]
spl_vec = CubicSplineInterpolation(xspl, yspl)
spl_vec(x)

##


@info("Stupid way")
@btime ($spl_tup)($x)
@info("SVector-spline way")
@btime ($spl_vec)($x)

