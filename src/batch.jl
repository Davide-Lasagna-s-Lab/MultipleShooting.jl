# Batches of states. The package does not know what a batch is: it is whatever the caller hands
# over, the seeds of the segments or the vectors attached to them, and the propagator of the
# caller advances it. The package only moves the states of a batch about, with the few functions
# of this file, and combines batches by broadcasting.
#
# The functions are written for an array whose last dimension runs over the states, so that a
# state may be a vector, a field on a grid or an array on a GPU: a matrix of columns, or an array
# type of the caller's with `similar`, `copy`, `copyto!`, `view`, `reverse!` and `randn!`. A batch
# of another kind, such as a vector of states advanced one at a time, gives its own methods of
# these functions, and broadcasts state by state.

"""
    nsegments(X)

The number of states of a batch, the size of its last dimension for an array.
"""
nsegments(X) = size(X, ndims(X))

"""
    dots(X, Y) -> c

The inner products of the states of the batches `X` and `Y` one by one, real, in a form that
broadcasts against a batch: for an array, an array whose last dimension runs over the states and
the others have size one, on the device of `X`. The Euclidean product of the numbers of a state by
default, the real part for complex ones. The caller gives a method for the type of its batches
when the states have another product: GMRES, the phase conditions, the bases of the subspace
iteration and the preconditioner all use it, so that they share one geometry.
"""
dots(X, Y) = sum(real.(conj.(X) .* Y); dims=1:ndims(X) - 1)

# the inner product of two batches, the sum over their states
_dot(X, Y) = sum(dots(X, Y))

# the states r of a batch, as a batch, without a copy
_states(X, r::AbstractUnitRange) = selectdim(X, ndims(X), r)

# the first and the last state of a batch, as a batch of one state: the first seed, and the end
# of the last segment
_head(X) = _states(X, 1:1)
_tail(X) = _states(X, nsegments(X):nsegments(X))

# block a of a batch made of blocks of m states, (a - 1)m + 1 to am
_block(X, a::Int, m::Int) = _states(X, ((a - 1)*m + 1):(a*m))

# a batch like X with b states, its entries not set, of the type that `similar` gives for the new
# size, which a propagator must be able to advance
_similar(X, b::Int) = similar(X, eltype(X), (Base.front(size(X))..., b))

# k′ copies of the batch X, one after the other, as one batch of k′ blocks
function _copies(X, k′::Int)
    m   = nsegments(X)
    Xk′ = _similar(X, k′*m)
    for a in 1:k′
        copyto!(_block(Xk′, a, m), X)
    end

    return Xk′
end

# coefficients one per state, from a vector on the host, in the form of dots, so that they
# broadcast against a batch, on its device
_coefficients(X, c::AbstractVector) =
    copyto!(similar(X, real(eltype(X)), ntuple(_ -> 1, ndims(X) - 1)..., length(c)), c)

# The states of X moved forward by one slot, forward in time along the orbit, cyclically and in
# place: state i goes into slot i + 1, and the last state into slot 1,
#
#     X = (x₁, x₂, x₃)   →   X = (x₃, x₁, x₂)
#
# In multiple shooting this puts the end of each segment in the row of the seed it has to reach.
# By two reversals along the last dimension, all of it, then all but the first slot, each in
# place; with five states,
#
#     X                                  (x₁, x₂, x₃, x₄, x₅)
#     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
#     slots 2 to 5 reversed              (x₅, x₁, x₂, x₃, x₄)
#
# the last state comes first, and the others keep their order one slot later
function _shift_forward!(X)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(_states(X, 2:nsegments(X)); dims=d)

    return X
end

# The states of X moved back by one slot, backward in time, the inverse of _shift_forward!: state
# i goes into slot i - 1, and the first state into the last slot,
#
#     X = (x₁, x₂, x₃)   →   X = (x₂, x₃, x₁)
#
# by all of it reversed, then all but the last slot,
#
#     X                                  (x₁, x₂, x₃, x₄, x₅)
#     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
#     slots 1 to 4 reversed              (x₂, x₃, x₄, x₅, x₁)
function _shift_backward!(X)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(_states(X, 1:nsegments(X)-1); dims=d)

    return X
end
