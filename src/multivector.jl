# The unknowns of multiple shooting: the seeds u₁, …, u_m, a batch of states, and the scalars, the
# period T and one shift per continuous symmetry. They form one vector, with the inner product
#
#     ⟨z, w⟩ = Σᵢ ⟨uᵢ, vᵢ⟩ + Σⱼ dⱼ d'ⱼ.
#
# Broadcasting acts on the two parts separately: an expression such as w .= α .* z .+ β .* w is
# split into w.X .= α .* z.X .+ β .* w.X on the seeds and w.d .= α .* z.d .+ β .* w.d on the
# scalars. The seeds broadcast as one array in the batched layout, on whatever device they live,
# and state by state in the sliced one.

"""
    MultiVector(X, d)

The unknowns of multiple shooting, the seeds and the scalars,

    z = (u₁, …, u_m, T, s₁, …, s_c),

with the inner product of the flattened vectors, ⟨z, w⟩ = Σᵢ ⟨uᵢ, wᵢ⟩ + Σₖ dₖ d′ₖ over the
scalars. It broadcasts like a vector, `w .= α .* z .+ w`, on the seeds and the scalars
separately.

# Arguments
- `X`: the seeds, a batch of m states, [`Batched`](@ref) or [`Sliced`](@ref).
- `d::Vector{Float64}`: the scalars `[T, s₁, s₂, …]`, the period followed by one shift per
  continuous symmetry, `[T]` for a periodic orbit. [`period`](@ref), [`nshifts`](@ref) and
  [`shift`](@ref) read them.
"""
struct MultiVector{A}
    X::A               # the seeds, a batch of states
    d::Vector{Float64} # the scalars: the period, then the shifts
end

nsegments(z::MultiVector) = nsegments(z.X)

# ---- the scalars: d[1] is the period, d[2:end] the shifts ----
"""
    period(z::MultiVector) -> T

The period, the first scalar of `z`.

# Arguments
- `z::MultiVector`: the unknowns.
"""
period(z::MultiVector) = z.d[1]

"""
    nshifts(z::MultiVector) -> Int

The number of shifts, one per continuous symmetry, zero for a periodic orbit.

# Arguments
- `z::MultiVector`: the unknowns.
"""
nshifts(z::MultiVector) = length(z.d) - 1

"""
    shift(z::MultiVector, i) -> sᵢ

The `i`-th shift.

# Arguments
- `z::MultiVector`: the unknowns.
- `i::Integer`: the index of the shift, from 1 to [`nshifts`](@ref)`(z)`, checked, since `i = 0`
  would return the period.
"""
function shift(z::MultiVector, i::Integer)
    1 <= i <= nshifts(z) ||
        throw(BoundsError(z.d[2:end], i))

    return z.d[1 + i]
end

# all the shifts, as a view
_shifts(z::MultiVector) = view(z.d, 2:lastindex(z.d))

Base.similar(z::MultiVector) = MultiVector(_similar(z.X), similar(z.d))

Base.copy(z::MultiVector) = MultiVector(_copy(z.X), copy(z.d))

LinearAlgebra.dot(z::MultiVector, w::MultiVector) = _dot(z.X, w.X) + LinearAlgebra.dot(z.d, w.d)

LinearAlgebra.norm(z::MultiVector) = sqrt(LinearAlgebra.dot(z, z))


# ---------------------------------------------------------------------------- #
# broadcasting                                                                 #
# ---------------------------------------------------------------------------- #

# a style of its own, which wins over the numbers of an expression
struct MultiStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:MultiVector}) = MultiStyle()
Base.BroadcastStyle(s::MultiStyle, ::Broadcast.DefaultArrayStyle{0}) = s
Base.broadcastable(z::MultiVector) = z

# the same expression on one part, :X or :d, or on state i of the seeds: every MultiVector
# replaced by that part
_part(z::MultiVector, part::Symbol)         = getfield(z, part)
_part(z::MultiVector, part::Symbol, i::Int) = getfield(z, part)[i]
_part(bc::Broadcast.Broadcasted, part::Symbol, i...) =
    Broadcast.broadcasted(bc.f, map(a -> _part(a, part, i...), bc.args)...)
_part(x, part::Symbol, i...) = x

# the expression on the seeds: on the whole array, or state by state
_materialize!(X::Batched, bc) = Broadcast.materialize!(X, _part(bc, :X))

function _materialize!(X::Sliced, bc)
    for i in eachindex(X)
        Broadcast.materialize!(X[i], _part(bc, :X, i))
    end

    return X
end

# w .= expression: the expression on the seeds, then on the scalars. Taking over materialize!
# and copyto! avoids the axes of the destination, which a MultiVector does not have
Broadcast.materialize!(::MultiStyle, w::MultiVector, bc::Broadcast.Broadcasted) = copyto!(w, bc)

function Base.copyto!(w::MultiVector, bc::Broadcast.Broadcasted)
    _materialize!(w.X, bc)
    Broadcast.materialize!(w.d, _part(bc, :d))

    return w
end

# z .+ w, out of place: a new vector shaped as the first MultiVector of the expression
Broadcast.instantiate(bc::Broadcast.Broadcasted{MultiStyle}) = bc
Base.copy(bc::Broadcast.Broadcasted{MultiStyle}) = copyto!(similar(_first(bc.args...)), bc)

_first(z::MultiVector, args...) = z
_first(bc::Broadcast.Broadcasted, args...) = something(_first(bc.args...), _first(args...))
_first(x, args...) = _first(args...)
_first() = nothing
