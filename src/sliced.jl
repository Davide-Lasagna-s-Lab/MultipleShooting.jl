# The operations of the package on a batch in the sliced layout, an AbstractSeeds whose array is a
# vector of states, each an array of any shape (see seeds.jl): each state by state.

# the Euclidean product of the states of two vectors of states, the real part for complex ones: a
# vector
_euclidean(A::AbstractVector{<:AbstractArray}, B::AbstractVector{<:AbstractArray}) =
    map((x, y) -> real(LinearAlgebra.dot(x, y)), A, B)

# a batch like X with its entries not set, each state like that of X, and a copy
_similar(X::Sliced{M, L}) where {M, L} = rewrap(X, map(similar, parent(X)), L)
_copy(X::Sliced{M, L}) where {M, L}    = rewrap(X, map(copy, parent(X)), L)

# a batch like X of M′ seeds in L′ blocks, its entries not set, the states shaped as the first of X
_similar(X::Sliced, M′::Int, L′::Int) =
    rewrap(X, [similar(first(parent(X))) for _ in 1:M′*L′], L′)

# random entries
_randn!(X::Sliced) = (foreach(Random.randn!, parent(X)); X)

# coefficients one per state, from a vector on the host, in the shape of dots, a vector
_coefficients(X::Sliced, c::AbstractVector) = collect(Float64, c)

# dest ← src, the data of the states: copying a vector of states with .= would copy the references
_copyto!(dest::Sliced, src::Sliced) = (foreach((d, s) -> d .= s, parent(dest), parent(src)); dest)

# dest ← f.(srcs...), state by state
function _map!(f, dest::Sliced, srcs::Sliced...)
    for i in eachindex(parent(dest))
        parent(dest)[i] .= f.(getindex.(parent.(srcs), i)...)
    end

    return dest
end

# the states of X moved forward by one slot, forward in time, and back by one, backward in time,
# cyclically, as those of batched.jl: by a permutation of the states, which moves no data
_shift_forward!(X::Sliced)  = (circshift!(parent(X), 1); X)
_shift_backward!(X::Sliced) = (circshift!(parent(X), -1); X)
