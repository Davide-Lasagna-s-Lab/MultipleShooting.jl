# The unknowns of multiple shooting: the seeds u₁, …, u_m, a batch of states, and the scalars, the
# period T and one shift per continuous symmetry. They form one vector, with the inner product
#
#     ⟨z, w⟩ = Σᵢ ⟨uᵢ, vᵢ⟩ + Σⱼ dⱼ d'ⱼ,
#
# where ⟨uᵢ, vᵢ⟩ is the inner product of two states. The vector carries it, as a function that
# gives the products of the states of two batches one by one, the Euclidean one unless another is
# given: GMRES, the phase conditions, the bases of the subspace iteration and the preconditioner
# all take it from there, so that they share one geometry.
#
# Broadcasting acts on the two parts separately: an expression such as w .= α .* z .+ β .* w is
# split into w.X .= α .* z.X .+ β .* w.X on the seeds and w.d .= α .* z.d .+ β .* w.d on the
# scalars. The seeds broadcast as one array in the batched layout, on whatever device they live,
# and state by state in the sliced one.

"""
    MultiVector(X, d; τ=nothing, dots=nothing)

The unknowns of multiple shooting, the seeds and the scalars,

    z = (u₁, …, u_m, T, s₁, …, s_c),

with the inner product ⟨z, w⟩ = Σᵢ ⟨uᵢ, wᵢ⟩ + Σₖ dₖ d′ₖ, the sum of those of the states and of
the scalars. It broadcasts like a vector, `w .= α .* z .+ w`, on the seeds and the scalars
separately.

# Arguments
- `X`: the seeds, a batch of m states, [`Batched`](@ref) or [`Sliced`](@ref).
- `d::Vector{Float64}`: the scalars `[T, s₁, s₂, …]`, the period followed by one shift per
  continuous symmetry, `[T]` for a periodic orbit. [`period`](@ref), [`nshifts`](@ref) and
  [`shift`](@ref) read them.
- `τ=nothing`: the lengths of the segments, one per seed, positive and adding up to the period.
  Equal segments, T/m, when not given. [`times`](@ref) reads them. The vector keeps their
  fractions of the period, so that a new period changes them all in proportion.
- `dots=nothing`: the inner product of the states, a function `(X, Y) -> c` of two batches that
  gives the products of their states one by one, real: for [`Batched`](@ref) an array whose last
  dimension runs over the states and the others have size one, so that it broadcasts against a
  batch, and for [`Sliced`](@ref) a vector. The Euclidean product of the numbers of a state when
  not given, the real part for complex ones. Every vector made from this one keeps it.
"""
struct MultiVector{A, D}
       X::A               # the seeds, a batch of states
       d::Vector{Float64} # the scalars: the period, then the shifts
       w::Vector{Float64} # the fractions of the period of the segments, τᵢ/T, not part of the vector
    dots::D               # the inner products of the states of two batches, one per state

    function MultiVector(X, d::AbstractVector; τ=nothing, dots=nothing)
        m = nsegments(X)
        d = convert(Vector{Float64}, d)

        # ---- input checks ----
        τ === nothing || length(τ) == m ||
            throw(ArgumentError("one time per segment is needed, got $(length(τ)) for $(m)"))
        τ === nothing || (all(>(0), τ) && sum(τ) ≈ d[1]) ||
            throw(ArgumentError("the times must be positive and add up to the period"))

        # ---- the fractions: equal segments unless the times are given ----
        w = τ === nothing ? fill(1/m, m) : collect(Float64, τ) ./ sum(τ)

        # ---- the inner product of the states: the Euclidean one unless another is given ----
        dots = something(dots, _dots)

        return new{typeof(X), typeof(dots)}(X, d, w, dots)
    end

    # a vector with the fractions of another, shared and not copied, and its inner product
    MultiVector(X, d::Vector{Float64}, w::Vector{Float64}, dots=_dots) =
        new{typeof(X), typeof(dots)}(X, d, w, dots)
end

"""
    times(z::MultiVector) -> τ

The lengths of the segments, τᵢ = wᵢ T, one per seed. They add up to the period. A change of the
period changes them all in proportion.

# Arguments
- `z::MultiVector`: the unknowns.
"""
times(z::MultiVector) = z.w .* period(z)

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

Base.similar(z::MultiVector) = MultiVector(_similar(z.X), similar(z.d), z.w, z.dots)

Base.copy(z::MultiVector) = MultiVector(_copy(z.X), copy(z.d), z.w, z.dots)

# the inner product of two batches of states like the seeds of z, the sum over their states
_dot(z::MultiVector, X, Y) = sum(z.dots(X, Y))

LinearAlgebra.dot(z::MultiVector, w::MultiVector) = _dot(z, z.X, w.X) + LinearAlgebra.dot(z.d, w.d)

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
