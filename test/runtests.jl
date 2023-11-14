using UltraFastACE
using Test

@testset "UltraFastACE.jl" begin
    # Write your tests here.
    @testset "Import ACE1" begin; include("test_import_ace1.jl"); end
end
