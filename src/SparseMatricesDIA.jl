module SparseMatricesDIA

using SparseArrays
using LinearAlgebra

import Base: size, getindex, setindex!, *
import LinearAlgebra: mul!
import SparseArrays: sparse

export SparseMatrixDIA

include("SparseMatrixDIA.jl")
include("SymSparseMatrixDIA.jl")

end
