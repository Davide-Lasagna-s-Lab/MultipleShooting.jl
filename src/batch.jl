# Batches of states. A batch holds the states that the propagator of the caller advances
# together, the seeds of the segments or the vectors attached to them. The package only moves the
# states of a batch about, and combines batches by broadcasting. A batch is
#
#     an array of numbers whose last dimension runs over the states, a Batched, so that a state
#     may be a vector, a field on a grid or an array on a GPU: a matrix of columns, or an array
#     type of the caller's, with `view` and `reverse!`; or
#
#     a type of the caller's that is not such an array, such as a vector of states advanced one
#     at a time, with its own methods of the public functions of this file, the first part, and
#     broadcasting state by state.
#
# Nothing else is a batch: the public functions have methods for a Batched only, and a type
# without its own fails at the first call. They are
#
#     nstates(X)              the number of states
#     states(X, r)            the states r, as a batch, without a copy
#     similar_states(X, b)    a batch like X with b states
#     roll_forward!(X)        the states moved forward by one slot, cyclically
#     roll_backward!(X)       and back by one
#     dots(X, Y)              the inner products of the states, one per state
#     coefficients(X, c)      coefficients one per state, in the form of dots
#
# and, of Base, `similar`, `copy`, `copyto!`, `randn!` and broadcasting. The second part holds the
# functions of the package's own, with an underscore, built on the public ones: no batch has to
# give them.


# ---------------------------------------------------------------------------- #
# public: what a batch gives                                                   #
# ---------------------------------------------------------------------------- #

"""
    Batched

A batch that is an array of numbers, with the states along its last dimension: the columns of a
matrix, the fields of a three-dimensional array. The methods of this file are for it, and a
propagator dispatches on it, or on an array type of its own.
"""
const Batched = AbstractArray{<:Number}

"""
    nstates(X)

The number of states of the batch `X`: for an array, the size of its last dimension.
"""
nstates(X::Batched) = size(X, ndims(X))

"""
    states(X, r) -> Y

The states `r`, a range, of the batch `X`, as a batch that shares them, without a copy.
"""
states(X::Batched, r::AbstractUnitRange) = selectdim(X, ndims(X), r)

"""
    similar_states(X, b) -> Y

A batch like `X` with `b` states, its entries not set: for an array, of the type that `similar`
gives for the new size, which the propagator must be able to advance.
"""
similar_states(X::Batched, b::Int) = similar(X, eltype(X), (Base.front(size(X))..., b))

"""
    roll_forward!(X) -> X

Move the states of the batch `X` forward by one slot, forward in time along the orbit, cyclically
and in place: state i goes into slot i + 1, and the last state into slot 1,

    X = (x₁, x₂, x₃)   →   X = (x₃, x₁, x₂)

In multiple shooting this puts the end of each segment in the row of the seed it has to reach.
"""
function roll_forward!(X::Batched)
    # By two reversals along the last dimension, all of it, then all but the first slot, each in
    # place; with five states,
    #
    #     X                                  (x₁, x₂, x₃, x₄, x₅)
    #     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
    #     slots 2 to 5 reversed              (x₅, x₁, x₂, x₃, x₄)
    #
    # the last state comes first, and the others keep their order one slot later
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(states(X, 2:nstates(X)); dims=d)

    return X
end

"""
    roll_backward!(X) -> X

Move the states of the batch `X` back by one slot, backward in time, cyclically and in place, the
inverse of [`roll_forward!`](@ref): state i goes into slot i - 1, and the first state into the
last slot,

    X = (x₁, x₂, x₃)   →   X = (x₂, x₃, x₁)
"""
function roll_backward!(X::Batched)
    # all of it reversed, then all but the last slot,
    #
    #     X                                  (x₁, x₂, x₃, x₄, x₅)
    #     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
    #     slots 1 to 4 reversed              (x₂, x₃, x₄, x₅, x₁)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(states(X, 1:nstates(X)-1); dims=d)

    return X
end

"""
    dots(X, Y) -> c

The inner products of the states of the batches `X` and `Y` one by one, real, in a form that
broadcasts against a batch: for an array, an array whose last dimension runs over the states and
the others have size one, on the device of `X`. The Euclidean product of the numbers of a state by
default, the real part for complex ones. The caller gives a method for the type of its batches
when the states have another product: GMRES, the phase conditions, the bases of the subspace
iteration and the preconditioner all use it, so that they share one geometry.
"""
dots(X::Batched, Y::Batched) = sum(real.(conj.(X) .* Y); dims=1:ndims(X) - 1)

"""
    coefficients(X, c) -> a

The coefficients of the vector `c`, one per state of the batch `X`, in the form of [`dots`](@ref),
so that `a .* X` is every state times its own: for an array, on its device.
"""
coefficients(X::Batched, c::AbstractVector) =
    copyto!(similar(X, real(eltype(X)), ntuple(_ -> 1, ndims(X) - 1)..., length(c)), c)


# ---------------------------------------------------------------------------- #
# internal: built on the public functions                                      #
# ---------------------------------------------------------------------------- #

# the inner product of two batches, the sum over their states
_dot(X, Y) = sum(dots(X, Y))

# the first and the last state of a batch, as a batch of one state: the first seed, and the end
# of the last segment
_head(X) = states(X, 1:1)
_tail(X) = states(X, nstates(X):nstates(X))
