# A batch that is not an array, for the tests: a vector of states, each an array of its own,
# advanced one at a time, with weights on the coordinates of a state in its inner product. It
# gives the methods that MultipleShooting asks of a batch (src/batch.jl) and those of Base that
# the package calls on one, and it broadcasts state by state.
import Random

struct Sliced{V<:AbstractVector, Q}
    states::V # the states, each an array
         q::Q # the weights of the coordinates in the inner product, 1 for the Euclidean one
end

Sliced(states) = Sliced(states, 1)

# ---- the states, one by one ----
Base.length(X::Sliced)           = length(X.states)
Base.getindex(X::Sliced, i::Int) = X.states[i]
Base.iterate(X::Sliced, s...)    = iterate(X.states, s...)

# ---- what MultipleShooting asks of a batch ----
MultipleShooting.nsegments(X::Sliced) = length(X.states)

MultipleShooting.dots(X::Sliced, Y::Sliced) =
    map((x, y) -> sum(X.q .* real.(conj.(x) .* y)), X.states, Y.states)

MultipleShooting._states(X::Sliced, r::AbstractUnitRange) = Sliced(view(X.states, r), X.q)

MultipleShooting._similar(X::Sliced, b::Int) = Sliced([similar(first(X.states)) for _ in 1:b], X.q)

MultipleShooting._coefficients(X::Sliced, c::AbstractVector) = collect(Float64, c)

MultipleShooting._shift_forward!(X::Sliced)  = (circshift!(X.states, 1); X)
MultipleShooting._shift_backward!(X::Sliced) = (circshift!(X.states, -1); X)

# ---- what Base gives for an array ----
Base.similar(X::Sliced)            = Sliced(map(similar, X.states), X.q)
Base.copy(X::Sliced)               = Sliced(map(copy, X.states), X.q)
Base.copyto!(X::Sliced, Y::Sliced) = (foreach(copyto!, X.states, Y.states); X)
Random.randn!(X::Sliced)           = (foreach(Random.randn!, X.states); X)

# ---- broadcasting, state by state: in the expression every batch stands for its state i, and
#      every vector of coefficients, one per state, for its entry i ----
struct SlicedStyle <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:Sliced}) = SlicedStyle()
Base.BroadcastStyle(s::SlicedStyle, ::Base.Broadcast.DefaultArrayStyle) = s
Base.Broadcast.broadcastable(X::Sliced) = X

_at(X::Sliced, i::Int)                = X.states[i]
_at(c::AbstractVector{<:Real}, i::Int) = c[i]
_at(bc::Base.Broadcast.Broadcasted, i::Int) =
    Base.Broadcast.broadcasted(bc.f, map(a -> _at(a, i), bc.args)...)
_at(x, i::Int) = x

function Base.Broadcast.materialize!(X::Sliced, bc::Base.Broadcast.Broadcasted)
    for i in eachindex(X.states)
        X.states[i] .= _at(bc, i)
    end

    return X
end
