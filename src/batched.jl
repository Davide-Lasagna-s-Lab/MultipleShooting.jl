# The operations of the package on a batch in the batched layout, an AbstractSeeds whose array is
# an array of numbers with the states along its last dimension (see seeds.jl): each on the whole
# array at once, on the device of the array.

# the Euclidean product of the states of two arrays, the real part for complex ones: an array
# whose last dimension runs over the states and the others have size one
_euclidean(A::AbstractArray{<:Number}, B::AbstractArray{<:Number}) =
    sum(real.(conj.(A) .* B); dims=1:ndims(A) - 1)

# a batch like X with its entries not set, of the array type that similar gives, and a copy
_similar(X::Batched{M, K}) where {M, K} = rewrap(X, similar(parent(X)), K)
_copy(X::Batched{M, K}) where {M, K}    = rewrap(X, copy(parent(X)), K)

# a batch like X of M′ seeds in K′ blocks, its entries not set, of the array type that similar
# gives for the new size, which a propagator must be able to advance
_similar(X::Batched, M′::Int, K′::Int) =
    rewrap(X, similar(parent(X), eltype(X), (Base.front(size(X))..., M′*K′)), K′)

# random entries
_randn!(X::Batched) = (Random.randn!(parent(X)); X)

# coefficients one per state, from a vector on the host, in the shape of dots, so that they
# broadcast against the array of a batch, on its device
_coefficients(X::Batched, c::AbstractVector) =
    copyto!(similar(parent(X), real(eltype(X)), ntuple(_ -> 1, ndims(X) - 1)..., length(c)), c)

# dest ← src, the data of the states
_copyto!(dest::Batched, src::Batched) = (copyto!(parent(dest), parent(src)); dest)

# dest ← f.(srcs...), on the whole array
_map!(f, dest::Batched, srcs::Batched...) = (parent(dest) .= f.(parent.(srcs)...); dest)

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
function _shift_forward!(X::Batched)
    A = parent(X)
    d = ndims(A)
    reverse!(A; dims=d)
    reverse!(selectdim(A, d, 2:size(A, d)); dims=d)

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
function _shift_backward!(X::Batched)
    A = parent(X)
    d = ndims(A)
    reverse!(A; dims=d)
    reverse!(selectdim(A, d, 1:size(A, d)-1); dims=d)

    return X
end
