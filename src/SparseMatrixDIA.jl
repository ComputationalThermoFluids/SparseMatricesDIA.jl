"""
    SparseMatrixDIA{T,I,V<:AbstractMatrix{T},D<:AbstractVector{I}} <: AbstractSparseMatrix{T,I}

Matrix type for storing square diagonal matrices.

See also

- <https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/appendices/mkl_appA_SMSF.htm>

"""
struct SparseMatrixDIA{T,I,V<:AbstractMatrix{T},D<:AbstractVector{I}} <: AbstractSparseMatrix{T,I}
    vals::V
    dist::D

    # also need to check
    # - elements of distance are unique
    # - elements of n
    function SparseMatrixDIA(vals::V, dist::D) where {V,D}
        if size(vals, 2) ≠ size(dist, 1)
            error("invalid multi-diagonal matrix")
        end
        new{eltype(V),eltype(D),V,D}(vals, dist)
    end
end

values(A::SparseMatrixDIA) = A.vals
distance(A::SparseMatrixDIA) = A.dist

size(A::SparseMatrixDIA) = (size(values(A), 1), size(values(A), 1))

function getindex(A::SparseMatrixDIA, i, j)
    vals = values(A)
    dist = distance(A)

    ind = findfirst(==(j-i), dist)

    isnothing(ind) && return zero(eltype(A))

    vals[i, ind]
end

function setindex!(A::SparseMatrixDIA, val, i, j)
    vals = values(A)
    dist = distance(A)

    ind = findfirst(==(j-i), dist)

    isnothing(ind) && return zero(eltype(A))

    vals[i, ind] = val
end

#lval(A::SparseMatrixDIA) = size(values(A), 1)
#ndiag(A::SparseMatrixDIA) = size(values(A), 2)

function *(A::SparseMatrixDIA{S}, x::AbstractVector{T}) where {S,T}
    y = similar(x, promote_type(S, T), size(A, 1))#, size(x))
    mul!(y, A, x)
    y
end

"""

    mul!(y::AbstractVector, A::SparseMatrixDIA, x::AbstractVector)

See also

- <http://www.netlib.org/utk/people/JackDongarra/etemplates/node383.html>

"""
function mul!(y::AbstractVector, A::SparseMatrixDIA, x::AbstractVector)
    vals = values(A)
    dist = distance(A)

    y .= zero(eltype(y))

    n = size(vals, 1)

    for (ind, d) in enumerate(dist)
        for j in max(1, 1-d):min(n, n-d)
            y[j] += vals[j, ind] * x[j+d]
        end
    end
end

function *(A::Transpose{S,<:SparseMatrixDIA}, x::AbstractVector{T}) where {S,T}
    y = zeros(promote_type(S, T), size(x))
    mul!(y, A, x)
end

function mul!(y::AbstractVector, A::Transpose{S,<:SparseMatrixDIA}, x::AbstractVector) where {S}
    vals = values(parent(A))
    dist = distance(parent(A))

    n = size(vals, 1)

    for (ind, d) in enumerate(dist)
        for j in max(1, 1+d):min(n, n+d)
            y[j] += vals[j-d, ind] * x[j-d]
        end
    end
end

"""
    sparse(A::SparseMatrixDIA)

Converts a `SparseMatrixDIA` to a `SparseMatrixCSC`.

"""
function sparse(A::SparseMatrixDIA)
    vals = values(A)
    dist = distance(A)

    n = size(vals, 1)

    pairs = map(enumerate(dist)) do (i, d)
        d => view(vals, max(1, 1-d):min(n, n-d), i)
    end

    spdiagm(pairs...)
end

"""
    SparseMatrixDIA(A::Transpose{T,<:SparseMatrixDIA}) where {T}

Build a `SparseMatrixDIA` from the lazy transpose of a `SparseMatrixDIA`.

"""
#=
function SparseMatrixDIA(A::Transpose{T,<:SparseMatrixDIA}) where {T}
end
=#

