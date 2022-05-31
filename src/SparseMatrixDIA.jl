"""
    SparseMatrixDIA{T,I,V<:AbstractMatrix{T},D<:AbstractVector{I}} <: AbstractSparseMatrix{T,I}

Matrix type for storing square diagonal matrices.

See also

- <https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/appendices/mkl_appA_SMSF.htm>

"""
struct SparseMatrixDIA{T,I,V<:AbstractMatrix{T},D<:AbstractVector{I}} <: AbstractSparseMatrix{T,I}
    values::V
    distance::D

    # also need to check
    # - elements of distance are unique
    # - elements of n
    function SparseMatrixDIA(values::V, distance::D) where {V,D}
        if size(values, 2) ≠ size(distance, 1)
            error("invalid multi-diagonal matrix")
        end
        new{eltype(V),eltype(D),V,D}(values, distance)
    end
end

values(A::SparseMatrixDIA) = A.values
distance(A::SparseMatrixDIA) = A.distance

size(A::SparseMatrixDIA) = (size(values(A), 1), size(values(A), 1))

#lval(A::SparseMatrixDIA) = size(values(A), 1)
#ndiag(A::SparseMatrixDIA) = size(values(A), 2)

function *(A::SparseMatrixDIA{S}, x::AbstractVector{T}) where {S,T}
    y = zeros(promote_type(S, T), size(x))
    mul!(y, A, x)
end

"""

    mul!(y::AbstractVector, A::SparseMatrixDIA, x::AbstractVector)

See also

- <http://www.netlib.org/utk/people/JackDongarra/etemplates/node383.html>

"""
function mul!(y::AbstractVector, A::SparseMatrixDIA, x::AbstractVector)
    vals = values(A)
    dist = distance(A)

    n = size(vals, 1)

    for (i, d) in enumerate(dist)
        for j in max(1, 1-d):min(n, n-d)
            y[j] += vals[j, i] * x[d + j]
        end
    end

    y
end

