using SparseMatricesDIA
using Test

const ⋅ = NaN

@testset "SparseMatricesDIA.jl" begin
    @testset "5x5" begin

        A = [ 1.; -2.;  0.; -4.;  0.;;
             -1.;  5.;  0.;  0.;  8.;;
             -3.;  0.;  4.;  2.;  0.;;
              0.;  0.;  6.;  7.;  0.;;
              0.;  0.;  4.;  0.; -5.]

        distance = [-3; -1;  0;  1;  2]

        values = [  ⋅;   ⋅;   ⋅; -4.;  8.;;
                    ⋅; -2.;  0.;  2.;  0.;;
                   1.;  5.;  4.;  7.; -5.;;
                  -1.;  0.;  6.;  0.;   ⋅;;
                  -3.;  0.;  4.;   ⋅;   ⋅]

        B = SparseMatrixDIA(values, distance)

        x = rand(5)

        @test iszero(A * x - B * x)
    end
end
