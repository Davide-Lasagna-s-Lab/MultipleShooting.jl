# The unknowns of multiple shooting: the seeds u₁, …, u_m, in whatever the caller holds them, and
# the scalars, the period T and one shift per continuous symmetry. They form one vector, with the inner product
#
#     ⟨z, w⟩ = Σᵢ ⟨uᵢ, vᵢ⟩ + Σⱼ dⱼ d'ⱼ,
#
# where ⟨uᵢ, vᵢ⟩ is the inner product of two states, that of the seeds, dots (see
# interface.jl): GMRES, the phase conditions, the bases of the subspace iteration and the
# preconditioner all take it from there, so that they share one geometry.
#
# Broadcasting acts on the two parts separately: an expression such as w .= α .* z .+ β .* w is
# split into w.X .= α .* z.X .+ β .* w.X on the seeds and w.d .= α .* z.d .+ β .* w.d on the
# scalars. The seeds broadcast as their type does, an array as one array, on whatever device it
# lives.

"""
    ShootingVector(X, d)

The unknowns of multiple shooting, the seeds and the scalars,

    z = (u₁, …, u_m, T, s₁, …, s_c),

with the inner product ⟨z, w⟩ = Σᵢ ⟨uᵢ, wᵢ⟩ + Σⱼ dⱼ d′ⱼ, the sum of those of the states,
[`dots`](@ref) of the seeds, and of the scalars. The m segments have the same length, T/m. It
broadcasts like a vector, `w .= α .* z .+ w`, on the seeds and the scalars separately.

# Arguments
- `X`: the seeds, m states, of a type with the methods that interface.jl asks of the seeds: an
  array of numbers, a [`Batched`](@ref), a [`Sliced`](@ref), or a type of the caller's.
- `d::Vector{Float64}`: the scalars `[T, s₁, s₂, …]`, the period followed by one shift per
  continuous symmetry, `[T]` for a periodic orbit. [`period`](@ref), [`nshifts`](@ref) and
  [`shift`](@ref) read them.
"""
struct ShootingVector{A}
    X::A               # the seeds
    d::Vector{Float64} # the scalars: the period, then the shifts

    ShootingVector(X, d::AbstractVector) = new{typeof(X)}(X, convert(Vector{Float64}, d))
end

# the number of segments of the orbit, that of the seeds
nsegments(z::ShootingVector) = nsegments(z.X)

# ---- the scalars: d[1] is the period, d[2:end] the shifts ----
"""
    period(z::ShootingVector) -> T

The period, the first scalar of `z`.

# Arguments
- `z::ShootingVector`: the unknowns.
"""
period(z::ShootingVector) = z.d[1]

"""
    nshifts(z::ShootingVector) -> Int

The number of shifts, one per continuous symmetry, zero for a periodic orbit.

# Arguments
- `z::ShootingVector`: the unknowns.
"""
nshifts(z::ShootingVector) = length(z.d) - 1

"""
    shift(z::ShootingVector, i) -> sᵢ

The `i`-th shift.

# Arguments
- `z::ShootingVector`: the unknowns.
- `i::Integer`: the index of the shift, from 1 to [`nshifts`](@ref)`(z)`, checked, since `i = 0`
  would return the period.
"""
function shift(z::ShootingVector, i::Integer)
    1 <= i <= nshifts(z) ||
        throw(BoundsError(z.d[2:end], i))

    return z.d[1 + i]
end

"""
    shifts(z::ShootingVector) -> s

All the shifts, a view of the scalars of `z`, empty for a periodic orbit: what the shift `S`
takes, as `S(X, shifts(z))`.

# Arguments
- `z::ShootingVector`: the unknowns.
"""
shifts(z::ShootingVector) = view(z.d, 2:lastindex(z.d))

Base.similar(z::ShootingVector) = ShootingVector(similar(z.X), similar(z.d))

Base.copy(z::ShootingVector) = ShootingVector(copy(z.X), copy(z.d))

LinearAlgebra.dot(z::ShootingVector, w::ShootingVector) =
    _dot(z.X, w.X) + LinearAlgebra.dot(z.d, w.d)

LinearAlgebra.norm(z::ShootingVector) = sqrt(LinearAlgebra.dot(z, z))


# ---------------------------------------------------------------------------- #
# broadcasting                                                                 #
# ---------------------------------------------------------------------------- #

# a style of its own, which wins over the numbers of an expression
struct ShootingStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:ShootingVector}) = ShootingStyle()
Base.BroadcastStyle(s::ShootingStyle, ::Broadcast.DefaultArrayStyle{0}) = s
Base.broadcastable(z::ShootingVector) = z

# the same expression on one part, :X or :d: every ShootingVector replaced by that part
_part(z::ShootingVector, part::Symbol) = getfield(z, part)
_part(bc::Broadcast.Broadcasted, part::Symbol) =
    Broadcast.broadcasted(bc.f, map(a -> _part(a, part), bc.args)...)
_part(x, part::Symbol) = x

# w .= expression: the expression on the seeds, then on the scalars. Taking over materialize!
# and copyto! avoids the axes of the destination, which a ShootingVector does not have
Broadcast.materialize!(::ShootingStyle, w::ShootingVector, bc::Broadcast.Broadcasted) =
    copyto!(w, bc)

function Base.copyto!(w::ShootingVector, bc::Broadcast.Broadcasted)
    Broadcast.materialize!(w.X, _part(bc, :X))
    Broadcast.materialize!(w.d, _part(bc, :d))

    return w
end

# z .+ w, out of place: a new vector shaped as the first ShootingVector of the expression
Broadcast.instantiate(bc::Broadcast.Broadcasted{ShootingStyle}) = bc
Base.copy(bc::Broadcast.Broadcasted{ShootingStyle}) = copyto!(similar(_first(bc.args...)), bc)

_first(z::ShootingVector, args...) = z
_first(bc::Broadcast.Broadcasted, args...) = something(_first(bc.args...), _first(args...))
_first(x, args...) = _first(args...)
_first() = nothing
