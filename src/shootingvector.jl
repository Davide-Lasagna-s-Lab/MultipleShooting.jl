# The unknowns of multiple shooting: the seeds u₁, …, u_m, in whatever the caller holds them, and
# scalars, the period T and one shift per continuous symmetry. They form one vector, with the inner
# product
#
#     ⟨z, z′⟩ = Σᵢ ⟨uᵢ, u′ᵢ⟩ + T T′ + Σⱼ sⱼ s′ⱼ,
#
# where ⟨uᵢ, u′ᵢ⟩ is the inner product of two states, that of the seeds, dots (see
# interface.jl): GMRES, the phase conditions, the bases of the subspace iteration and the
# preconditioner all take it from there, so that they share one geometry.
#
# Broadcasting acts on the two parts separately: an expression such as w .= α .* z .+ β .* w is
# split into w.X .= α .* z.X .+ β .* w.X on the seeds and w.d .= α .* z.d .+ β .* w.d on the
# scalars. The seeds broadcast as their type does, an array as one array, on whatever device it
# lives.

"""
    ShootingVector(X, d; w=nothing)

The unknowns of multiple shooting, the seeds and the scalars,

    z = (u₁, …, u_m, T, s₁, …, s_c),

with the inner product ⟨z, z′⟩ = Σᵢ ⟨uᵢ, u′ᵢ⟩ + T T′ + Σⱼ sⱼ s′ⱼ: the sum of the products of
the states, [`dots`](@ref) of the seeds, and the Euclidean one of the scalars, with no weight
between the two. The m segments have the same length, T/m, or, with `w`, the lengths wᵢ T, fixed
fractions of the period, so that a new period changes them all in proportion; [`times`](@ref)
gives them. `dot` takes the real part, `norm` is its square root. It broadcasts like a vector,
`y .= α .* z .+ y`, on the seeds and the scalars separately, and every vector made from it keeps
its fractions.

# Arguments
- `X`: the seeds, m states, of a type with the methods that interface.jl asks of the seeds: an
  array of numbers, a [`Batched`](@ref), a [`Sliced`](@ref), or a type of the caller's.
- `d::AbstractVector`: the scalars `[T, s₁, s₂, …]`, the period followed by one shift per
  continuous symmetry, `[T]` for a periodic orbit, converted to `Vector{Float64}`.
  [`period`](@ref), [`nshifts`](@ref) and [`shift`](@ref) read them.
- `w=nothing`: the fractions of the period of the segments, one per seed, positive and adding up to
  one, for segments of unequal lengths, as `nᵢ/N` for segments of nᵢ of the N steps of a
  recurrence; equal segments when not given.
"""
struct ShootingVector{A}
    X::A                                # the seeds
    d::Vector{Float64}                  # the scalars: the period, then the shifts
    w::Union{Nothing, Vector{Float64}}  # the fractions of the period of the segments, nothing
                                        # for equal ones; shared, not copied, by the vectors
                                        # made from this one

    function ShootingVector(X, d::AbstractVector; w=nothing)
        # ---- input checks ----
        w === nothing || length(w) == nsegments(X) ||
            throw(ArgumentError("one fraction per segment is needed, got $(length(w)) for " *
                                "$(nsegments(X))"))
        w === nothing || (all(>(0), w) && sum(w) ≈ 1) ||
            throw(ArgumentError("the fractions of the segments must be positive and add up to one"))

        return new{typeof(X)}(X,
                              convert(Vector{Float64}, d),
                              w === nothing ? nothing : convert(Vector{Float64}, w))
    end
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
takes, one number each, as `S(X, shifts(z)...)`.

# Arguments
- `z::ShootingVector`: the unknowns.
"""
shifts(z::ShootingVector) = view(z.d, 2:lastindex(z.d))

"""
    times(z::ShootingVector) -> τ

The lengths of the segments, what the flow and the tangent take: the number T/m for equal
segments, or the vector of the lengths wᵢ T for segments of unequal lengths, which add up to the
period. A change of the period changes them all in proportion.

# Arguments
- `z::ShootingVector`: the unknowns.
"""
times(z::ShootingVector) = z.w === nothing ? period(z)/nsegments(z) : z.w .* period(z)

# similar and copy act on the seeds and on the scalars, and keep the fractions; copy is deep
Base.similar(z::ShootingVector) = ShootingVector(similar(z.X), similar(z.d); w=z.w)

Base.copy(z::ShootingVector) = ShootingVector(copy(z.X), copy(z.d); w=z.w)

# the inner product of the seeds, of their type, the real part for an array of complex states,
# and the Euclidean one of the scalars
LinearAlgebra.dot(z::ShootingVector, w::ShootingVector) =
    real(LinearAlgebra.dot(z.X, w.X)) + LinearAlgebra.dot(z.d, w.d)

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
