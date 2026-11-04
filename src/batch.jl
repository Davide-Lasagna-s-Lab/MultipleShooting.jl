# Batches of states, in two layouts:
#
#     batched   an array of numbers whose last dimension runs over the states, so that a state
#               may be a vector, a field on a grid or an array on a GPU, and a propagator can
#               advance the whole array at once
#     sliced    a vector of states, each an array of any shape, advanced one at a time
#
# Besides broadcasting, these are the only operations the code makes on a batch.

const Batched = AbstractArray{<:Number}
const Sliced  = AbstractVector{<:AbstractArray}

"""
    nsegments(X)

The number of states of the batch `X`: the size of its last dimension, batched, or its length,
sliced.
"""
nsegments(X::Batched) = size(X, ndims(X))
nsegments(X::Sliced)  = length(X)

# a copy, and a batch of the same layout with its entries not set
_copy(X::Batched) = copy(X)
_copy(X::Sliced)  = map(copy, X)

_similar(X::Batched) = similar(X)
_similar(X::Sliced)  = map(similar, X)

# the first and the last state of a batch, as batches of one state, without a copy
_head(X::Batched) = selectdim(X, ndims(X), 1:1)
_head(X::Sliced)  = view(X, 1:1)

_tail(X::Batched) = selectdim(X, ndims(X), size(X, ndims(X)):size(X, ndims(X)))
_tail(X::Sliced)  = view(X, lastindex(X):lastindex(X))

# the inner product of two batches, the sum over their states
_dot(X::Batched, Y::Batched) = dot(X, Y)
_dot(X::Sliced, Y::Sliced)   = sum(i -> dot(X[i], Y[i]), eachindex(X))

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
# Batched, by three reversals along the last dimension, the first of length one; sliced, by a
# permutation of the states, which moves no data
function _roll!(X::Batched)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(selectdim(X, d, 2:size(X, d)); dims=d)

    return X
end

_roll!(X::Sliced) = circshift!(X, 1)
