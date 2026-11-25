# Batches of states, in two layouts:
#
#     batched   an array of numbers whose last dimension runs over the states, so that a state
#               may be a vector, a field on a grid or an array on a GPU, and a propagator can
#               advance the whole array at once
#     sliced    a vector of states, each an array of any shape, advanced one at a time
#
# A batch is an AbstractSeeds{M, L}: the states at M seeds, the shooting points, L at each, in L
# blocks of M one after the other, block a the states (a - 1)M + 1 to aM,
#
#     X = (x₁¹, …, x_M¹, x₁², …, x_M², …, x₁ᴸ, …, x_Mᴸ).
#
# The seeds are a batch of one block, L = 1, and the k′ directions of a subspace step at every seed
# a batch of k′ blocks, block a direction a, as tangents! takes them. A batch wraps the array of
# its states, which parent gives, and broadcasts as that array does. Besides broadcasting, the
# operations of this file and of the two of the layouts, batched.jl and sliced.jl, are the only
# ones the code makes on a batch, all on that array.

"""
    AbstractSeeds{M, L, T, N} <: AbstractArray{T, N}

A batch of the states at `M` seeds, `L` at each, in `L` blocks of `M`: block a holds the states
(a - 1)M + 1 to aM, one at every seed. The seeds of multiple shooting are an
`AbstractSeeds{M, 1}`, and the k′ directions of a subspace step at every seed an
`AbstractSeeds{M, k′}`, so that the sizes are in the types, as in
`tangents!(P, X::AbstractSeeds{M, 1}, V::AbstractSeeds{M, L}, τ)`.

[`Seeds`](@ref) wraps any array. A package gives its own subtype for its states, with the methods

    parent(X)            the array of the states, batched or sliced
    rewrap(X, data, L)   a batch like X on the array data, of the same layout, in L blocks
    dots(X, Y)           the inner products of the states, one per state, Euclidean by default

and it broadcasts as its array, unless it gives its own broadcasting.
"""
abstract type AbstractSeeds{M, L, T, N} <: AbstractArray{T, N} end

"""
    Batched{M, L}

An [`AbstractSeeds`](@ref) in the batched layout: its array is an array of numbers whose last
dimension runs over the states. A propagator dispatches on it to advance all the states at once.
"""
const Batched{M, L} = AbstractSeeds{M, L, <:Number}

"""
    Sliced{M, L}

An [`AbstractSeeds`](@ref) in the sliced layout: its array is a vector of states, each an array
of any shape. A propagator dispatches on it to advance the states one at a time.
"""
const Sliced{M, L} = AbstractSeeds{M, L, <:AbstractArray, 1}

"""
    Seeds(data; L=1, dots=nothing)

The states in the array `data` as a batch of `L` blocks, an `AbstractSeeds{M, L}` with M the
number of states over L: [`Batched`](@ref) for an array of numbers whose last dimension runs over
the states, [`Sliced`](@ref) for a vector of states. It keeps `data`, not copied, which `parent`
returns, and broadcasts as it.

# Arguments
- `data`: the states.
- `L::Int=1`: the number of blocks, the states at every seed.
- `dots=nothing`: the inner product of the states, a function `(A, B) -> c` of the arrays of two
  batches that gives the products of their states one by one, real: for the batched layout an
  array whose last dimension runs over the states and the others have size one, so that it
  broadcasts against the array, and for the sliced one a vector. The Euclidean product of the
  numbers of a state when not given, the real part for complex ones. The batches made from this
  one keep it.
"""
struct Seeds{M, L, T, N, A<:AbstractArray{T, N}, D} <: AbstractSeeds{M, L, T, N}
    data::A # the states, along the last dimension of an array or in a vector
    dots::D # the inner products of the states of two arrays like data, one per state

    function Seeds(data::AbstractArray; L::Int=1, dots=nothing)
        n = _nstates(data)

        # ---- input checks ----
        n % L == 0 ||
            throw(ArgumentError("$(n) states do not make $(L) blocks"))

        # ---- the inner product: the Euclidean one unless another is given ----
        dots = something(dots, _euclidean)

        return new{n ÷ L, L, eltype(data), ndims(data), typeof(data), typeof(dots)}(data,
                                                                                    dots)
    end
end

"""
    nsegments(X)

The number of seeds of a batch, `M` of an [`AbstractSeeds`](@ref)`{M, L}`: the number of its
states when it is a batch of seeds, `L = 1`.
"""
nsegments(::AbstractSeeds{M}) where {M} = M

"""
    rewrap(X, data, L)

A batch like `X`, of the same type up to its sizes, on the array `data` of the same layout, in
`L` blocks: the method that a subtype of [`AbstractSeeds`](@ref) gives, by which the package makes
views, copies and new batches of the sizes it needs.
"""
rewrap(X::Seeds, data::AbstractArray, L::Int) = Seeds(data; L=L, dots=X.dots)

"""
    dots(X, Y) -> c

The inner products of the states of the batches `X` and `Y` one by one, real: batched, an array on
the device of `X` whose last dimension runs over the states and the others have size one, so that
it broadcasts against the array of a batch; sliced, a vector. The Euclidean product of the numbers
of a state by default, the real part for complex ones; for [`Seeds`](@ref) the product it was
given. A subtype of [`AbstractSeeds`](@ref) gives its own method for the product of its states.
"""
dots(X::AbstractSeeds, Y::AbstractSeeds) = _euclidean(parent(X), parent(Y))
dots(X::Seeds, Y::Seeds)                 = X.dots(parent(X), parent(Y))

# the Euclidean product of the states of two arrays of batches, the real part for complex ones,
# in batched.jl and sliced.jl
function _euclidean end

# the inner product of two batches, the sum over their states
_dot(X, Y) = sum(dots(X, Y))


# ---------------------------------------------------------------------------- #
# a batch as an array                                                          #
# ---------------------------------------------------------------------------- #

# the array of the states, and a batch indexed as it
Base.parent(X::Seeds) = X.data
Base.size(X::AbstractSeeds) = size(parent(X))
Base.IndexStyle(::Type{<:Seeds{M, L, T, N, A}}) where {M, L, T, N, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(X::AbstractSeeds, I::Int...) = parent(X)[I...]
Base.@propagate_inbounds Base.setindex!(X::AbstractSeeds, a, I::Int...) = (parent(X)[I...] = a)

# a batch like X and a copy of it; an array of another type or size on the device of X
Base.similar(X::AbstractSeeds) = _similar(X)
Base.copy(X::AbstractSeeds)    = _copy(X)
Base.similar(X::Seeds, ::Type{T}, dims::Dims) where {T} = similar(X.data, T, dims)

# the states of a batch from those of another, and all of them set to a
Base.copyto!(X::AbstractSeeds, Y::AbstractSeeds) = (copyto!(parent(X), parent(Y)); X)
Base.fill!(X::AbstractSeeds, a)                  = (fill!(parent(X), a); X)

# broadcasting: a batch in an expression stands for its array, and an expression assigned to a
# batch is written into its array, so that it runs on the device of the array
Base.Broadcast.broadcastable(X::AbstractSeeds) = parent(X)
Base.Broadcast.materialize!(X::AbstractSeeds, bc::Base.Broadcast.Broadcasted) =
    (Base.Broadcast.materialize!(parent(X), bc); X)


# ---------------------------------------------------------------------------- #
# the operations of the package on a batch                                     #
# ---------------------------------------------------------------------------- #

# the number of states of the array of a batch, along its last dimension, or of a vector of states
_nstates(data::AbstractArray) = size(data, ndims(data))

# the states r of a batch, a view, as a batch of one block
_states(X::AbstractSeeds, r::AbstractUnitRange) =
    rewrap(X, selectdim(parent(X), ndims(parent(X)), r), 1)

# block a of a batch, a view, the states (a - 1)M + 1 to aM, one at every seed
_block(X::AbstractSeeds{M}, a::Int) where {M} = _states(X, ((a - 1)*M + 1):(a*M))

# the first and the last state of a batch of seeds, as a batch of one state, a view: the first
# seed, and the end of the last segment
_head(X::AbstractSeeds{M, 1}) where {M} = _states(X, 1:1)
_tail(X::AbstractSeeds{M, 1}) where {M} = _states(X, M:M)

# L copies of the seeds X, one after the other, as one batch of L blocks
function _copies(X::AbstractSeeds{M, 1}, L::Int) where {M}
    XL = _similar(X, M, L)
    for a in 1:L
        _copyto!(_block(XL, a), X)
    end

    return XL
end
