# The seeds as an array of numbers: the methods of the functions that the package asks of the
# seeds (interface.jl) for an array whose last dimension runs over the states, a Batched, so that a
# state may be a vector, a field on a grid or an array on a GPU. A matrix of columns works as it
# is, and so does an array type of the caller's, with `view` and `reverse!`. Seeds of another
# kind, such as a vector of states advanced one at a time, have their own methods, written by the
# caller.

"""
    Batched

Seeds that are an array of numbers, with the states along its last dimension: the columns of a
matrix, the fields of a three-dimensional array. The package gives the methods of the interface
of the seeds for it, and a propagator dispatches on it, or on an array type of its own.
"""
const Batched = AbstractArray{<:Number}

# the size of the last dimension
nsegments(X::Batched) = size(X, ndims(X))

# a view along the last dimension
states(X::Batched, r::AbstractUnitRange) = selectdim(X, ndims(X), r)

# of the type that `similar` gives for the new size
similar_states(X::Batched, b::Int) = similar(X, eltype(X), (Base.front(size(X))..., b))

# By two reversals along the last dimension, all of it, then all but the first place, each in
# place; with five states,
#
#     X                                  (x₁, x₂, x₃, x₄, x₅)
#     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
#     places 2 to 5 reversed             (x₅, x₁, x₂, x₃, x₄)
#
# the last state comes first, and the others keep their order one place later
function roll_forward!(X::Batched)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(states(X, 2:nsegments(X)); dims=d)
    return X
end

# all of it reversed, then all but the last place,
#
#     X                                  (x₁, x₂, x₃, x₄, x₅)
#     all of it reversed                 (x₅, x₄, x₃, x₂, x₁)
#     places 1 to 4 reversed             (x₂, x₃, x₄, x₅, x₁)
function roll_backward!(X::Batched)
    d = ndims(X)
    reverse!(X; dims=d)
    reverse!(states(X, 1:nsegments(X)-1); dims=d)
    return X
end

# The Euclidean product of the numbers of a state, the real part for complex ones: an array whose
# last dimension runs over the states and the others have size one, on the device of X. An array
# type of the caller's gives its own method when its states have another product
dots(X::Batched, Y::Batched) = sum(real.(conj.(X) .* Y); dims=1:ndims(X) - 1)

# in the shape of dots, on the device of X
coefficients(X::Batched, c::AbstractVector) =
    copyto!(similar(X, real(eltype(X)), ntuple(_ -> 1, ndims(X) - 1)..., length(c)), c)
