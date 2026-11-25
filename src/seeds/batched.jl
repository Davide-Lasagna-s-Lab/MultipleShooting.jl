# The seeds as an array of numbers: the methods of the functions that the package asks of the
# seeds (interface.jl) for an array whose last dimension runs over the states, a Batched.
#
# What it is for. All the states sit in one array, so that the propagator advances them together,
# in one sweep over the array: the columns of a matrix for states that are vectors, an
# N_x × N_y × m array for fields on a grid, an array on a GPU. This is the kind to use whenever
# the states have one shape.
#
# Why an alias, and not a type. On an array of numbers Base already does what the package needs:
# `similar`, `copy`, `copyto!`, `randn!` and broadcasting act on the numbers of all the states at
# once, and a view along the last dimension is a batch of fewer states that shares them. Nothing
# has to be wrapped. A matrix is seeds as it is, and so is an array type of the caller's, as a
# type of fields on a GPU: it has the methods of this file for free, and needs only what its array
# type must give, `similar` for another size, `view` and `reverse!` along the last dimension. A
# wrapper would have to be put on and taken off at every call, and would give all of this up.
#
# What it costs. Every array of numbers is taken for seeds, and the convention, the states along
# the last dimension, is not checked: a matrix whose rows are the states gives wrong results, not
# an error.
#
# The inner product is the Euclidean one. States with another product are an array type of the
# caller's with its own method of `dots`.

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
