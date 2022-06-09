struct SymSparseMatrixDIA{T,I,V,D} <: AbstractSparseMatrix{T,I}
    upper::SparseMatrixDIA{T,I,V,D}
end

