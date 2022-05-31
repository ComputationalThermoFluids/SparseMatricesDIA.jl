module SparseMatricesDIA

using SparseArrays
using LinearAlgebra

import Base: size, *
import LinearAlgebra: mul!

export SparseMatrixDIA

include("SparseMatrixDIA.jl")

end
