# Batches of states, in two layouts:
#
#     batched   an array of numbers whose last dimension runs over the states, so that a state
#               may be a vector, a field on a grid or an array on a GPU, and a propagator can
#               advance the whole array at once
#     sliced    a vector of states, each an array of any shape, advanced one at a time
#
# The bases of the subspace iteration are k batches like the seeds, one per direction. Besides
# broadcasting, these are the only operations the code makes on a batch.

"""
    Batched

A batch in the batched layout: an array of numbers whose last dimension runs over the states. A
propagator dispatches on it to advance all the states at once.
"""
const Batched = AbstractArray{<:Number}

"""
    Sliced

A batch in the sliced layout: a vector of states, each an array of any shape. A propagator
dispatches on it to advance the states one at a time.
"""
const Sliced = AbstractVector{<:AbstractArray}

"""
    nsegments(X)

The number of states of a batch.

# Arguments
- `X`: a batch, [`Batched`](@ref), whose last dimension runs over the states, or
  [`Sliced`](@ref), a vector of states.
"""
nsegments(X::Batched) = size(X, ndims(X))
nsegments(X::Sliced)  = length(X)

# a copy, and a batch of the same layout with its entries not set, or random
_copy(X::Batched) = copy(X)
_copy(X::Sliced)  = map(copy, X)

_similar(X::Batched) = similar(X)
_similar(X::Sliced)  = map(similar, X)

_randn!(X::Batched) = Random.randn!(X)
_randn!(X::Sliced)  = (foreach(Random.randn!, X); X)

# the first and the last state of a batch, as a batch of one state, without a copy: the first
# seed, and the end of the last segment
_head(X::Batched) = selectdim(X, ndims(X), 1:1)
_head(X::Sliced)  = view(X, 1:1)

_tail(X::Batched) = selectdim(X, ndims(X), nsegments(X):nsegments(X))
_tail(X::Sliced)  = view(X, lastindex(X):lastindex(X))

# the inner product of two batches, the sum over their states
_dot(X::Batched, Y::Batched) = LinearAlgebra.dot(X, Y)
_dot(X::Sliced, Y::Sliced)   = sum(i -> LinearAlgebra.dot(X[i], Y[i]), eachindex(X))

# the inner products of the states of two batches, one per state: batched, an array whose last
# dimension runs over the states and the others have size one, so that it broadcasts against a
# batch; sliced, a vector
_dots(X::Batched, Y::Batched) = sum(X .* Y; dims=1:ndims(X) - 1)
_dots(X::Sliced, Y::Sliced)   = map(LinearAlgebra.dot, X, Y)

# coefficients one per state, from a vector on the host, in the shape of _dots, so that they
# broadcast against a batch: batched, on the device of X; sliced, a vector
_coefficients(X::Batched, c::AbstractVector) =
    copyto!(similar(X, real(eltype(X)), ntuple(_ -> 1, ndims(X) - 1)..., length(c)), c)
_coefficients(X::Sliced, c::AbstractVector) = collect(Float64, c)

# every state of X times its own coefficient, in place
_scale!(X::Batched, c::AbstractVector) = (X .*= _coefficients(X, c); X)
_scale!(X::Sliced, c::AbstractVector)  = (foreach((x, a) -> x .*= a, X, c); X)

# dest ← src, the data of the states: copying a vector of states with .= would copy the references
_copyto!(dest::Batched, src::Batched) = copyto!(dest, src)
_copyto!(dest::Sliced, src::Sliced)   = (foreach((d, s) -> d .= s, dest, src); dest)

# dest ← f.(srcs...), on the whole array or state by state
_map!(f, dest::Batched, srcs...) = (dest .= f.(srcs...); dest)

function _map!(f, dest::Sliced, srcs...)
    for i in eachindex(dest)
        dest[i] .= f.(getindex.(srcs, i)...)
    end

    return dest
end

# the states of X moved forward by one, cyclically and in place: state i goes into slot i + 1, and
# the last state into slot 1,
#
#     X = (x₁, x₂, x₃)   →   X = (x₃, x₁, x₂)
#
# In multiple shooting this puts the end of each segment in the row of the seed it has to reach.
# Batched, by two reversals along the last dimension, all of it, then all but the first slot, each
# in place; with five states,
#
#     X                                  (x₁, x₂, x₃, x₄, x₅)
#     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
#     slots 2 to 5 reversed              (x₅, x₁, x₂, x₃, x₄)
#
# the last state comes first, and the others keep their order one slot later. Sliced, by a
# permutation of the states, which moves no data
function _roll!(X::Batched)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(selectdim(X, d, 2:size(X, d)); dims=d)

    return X
end

_roll!(X::Sliced) = circshift!(X, 1)
