
using UltraFastACE, StaticArrays, BenchmarkTools
using UltraFastACE: generate_Zlms, Zlms
using StaticPolynomials: jacobian 

L = 5 
valL = Val(5)
val3 = Val(3)
Z5 = generate_Zlms(L)

xx = @SVector randn(3)

Z5(xx)

@btime ($Z5)($xx)
@btime ($Z5)($xx)
@btime jacobian($Z5, $xx)
@btime jacobian($Z5, $xx)


##


Zlms(valL, xx) - Z5(xx)

@btime Zlms($valL, $xx)
@btime ($Z5)($xx)

Zlms(val3, xx)
@btime Zlms($val3, $xx)