# The seeds as a vector of states: the methods of the functions that the package asks of the seeds
# (interface.jl) for a Sliced, which holds every state in an array of its own.
#
# What it is for. A flow that advances the states one at a time, each with its own number of
# steps or on its own task, or states that are not the slices of one array. The states may have
# any shape. With states of one shape and a flow that takes them all at once, an array, a
# Batched, is simpler and faster.
#
# Why a type, and not a plain vector of arrays. On a vector of arrays Base acts on the vector, that
# is on the references to the states, and not on the states:
#
#     similar(X)     gives a vector of references that are not set, with no state behind them
#     copy(X)        gives another vector of the same states, so that a copy changes with X
#     Y .= X         makes the states of Y those of X: in the residual, r .= z followed by the
#                    propagation of r would overwrite the unknowns, and r - z would vanish
#
# and a vector of m states could not be told from the vector of the k′ directions that the tangent
# takes. Sliced wraps the vector and gives each of these operations state by state, so that the
# package uses it as it uses an array. The price is the wrapper, `Sliced(states)` going in and
# `X[i]` or `X.states` coming out.
#
# Broadcasting is state by state: an expression assigned to a Sliced is evaluated once per state,
# with every Sliced in it standing for its state i, and every vector of coefficients, one per
# state as `dots` gives them, for its entry i.
#
# The inner product is that of two states, the Euclidean one unless `dot` is given. The seeds made
# from a Sliced keep it.

"""
    Sliced(states; dot=nothing)

Seeds that are a vector of states, each an array of its own, of any shape: `X[i]` is state i. The
package gives the methods of the interface of the seeds for it, and a flow dispatches on it
to advance the states one at a time. It keeps `states`, not copied, and it broadcasts state by
state: in an expression every `Sliced` stands for its state i, and every vector of coefficients,
one per state, for its entry i.

# Arguments
- `states::AbstractVector`: the states.
- `dot=nothing`: the inner product of two states, a function `(x, y) -> c` with `c` real. The
  Euclidean one when not given, the real part for complex states. The seeds made from these keep
  it.
"""
struct Sliced{V<:AbstractVector, D}
    states::V # the states, each an array of its own
       dot::D # the inner product of two states, real

    Sliced(states::AbstractVector; dot=nothing) =
        new{typeof(states), typeof(something(dot, _realdot))}(states, something(dot, _realdot))
end

# the Euclidean product of two states, the real part for complex ones
_realdot(x, y) = real(LinearAlgebra.dot(x, y))


# ---------------------------------------------------------------------------- #
# the interface of the seeds                                                   #
# ---------------------------------------------------------------------------- #

nsegments(X::Sliced) = length(X.states)

# a view of the vector of the states
states(X::Sliced, r::AbstractUnitRange) = Sliced(view(X.states, r); dot=X.dot)

# states shaped as the first of X
similar_states(X::Sliced, b::Int) = Sliced([similar(first(X.states)) for _ in 1:b]; dot=X.dot)

# by a permutation of the states, which moves no data
roll_forward!(X::Sliced)  = (circshift!(X.states,  1); X)
roll_backward!(X::Sliced) = (circshift!(X.states, -1); X)

# a vector, one product per state
dots(X::Sliced, Y::Sliced) = map(X.dot, X.states, Y.states)

coefficients(X::Sliced, c::AbstractVector) = collect(Float64, c)


# ---------------------------------------------------------------------------- #
# what Base gives for an array                                                 #
# ---------------------------------------------------------------------------- #

# the states, one by one
Base.length(X::Sliced)           = length(X.states)
Base.getindex(X::Sliced, i::Int) = X.states[i]
Base.iterate(X::Sliced, s...)    = iterate(X.states, s...)

# seeds like X and a copy, state by state, and the states of X from those of Y
Base.similar(X::Sliced)            = Sliced(map(similar, X.states); dot=X.dot)
Base.copy(X::Sliced)               = Sliced(map(copy, X.states); dot=X.dot)
Base.copyto!(X::Sliced, Y::Sliced) = (foreach(copyto!, X.states, Y.states); X)
Random.randn!(X::Sliced)           = (foreach(Random.randn!, X.states); X)

# Broadcasting, state by state: an expression assigned to a Sliced is evaluated once per state,
# with every Sliced in it replaced by its state i and every vector of coefficients by its entry i
struct SlicedStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:Sliced}) = SlicedStyle()
Base.BroadcastStyle(s::SlicedStyle, ::Broadcast.DefaultArrayStyle) = s
Broadcast.broadcastable(X::Sliced) = X

_state(X::Sliced, i::Int)                 = X.states[i]
_state(c::AbstractVector{<:Real}, i::Int) = c[i]
_state(bc::Broadcast.Broadcasted, i::Int) = Broadcast.broadcasted(bc.f, map(a -> _state(a, i), bc.args)...)
_state(x, i::Int)                         = x

function Broadcast.materialize!(X::Sliced, bc::Broadcast.Broadcasted)
    for i in eachindex(X.states)
        X.states[i] .= _state(bc, i)
    end
    return X
end
