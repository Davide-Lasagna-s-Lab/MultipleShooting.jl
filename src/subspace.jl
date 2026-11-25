# Subspace iteration around the loop, after Lust (1997): the bases of the leading directions of the
# tangent maps that the preconditioner needs at every shooting point (preconditioner.jl), in the
# order of the paper: the step, the Galerkin blocks, the turn and the test.
#
# Names. As in the paper, each basis holds k′ directions and the test checks the leading k ≤ k′ of
# them, the keywords `k′` and `k`. The shooting points are the seeds of the point z.
#
# The step. At every shooting point i an orthonormal basis Vᵢ of k′ directions. Step r carries the
# basis of every point over its segment by the tangent map and orthonormalises the images at the
# next point,
#
#     ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r),   i = 1, …, m - 1,   S(s) ψ_m V_m^(r-1) = V₁^(r) R_m^(r),
#
# a QR factorisation by two passes of modified Gram–Schmidt, with Rᵢ^(r) upper triangular. The k′
# directions of the m segments go to tangent! in one call, so that a step costs k′ integrations of
# the tangent equation over the orbit, and one of the equation when the propagator shares it among
# the directions. On a fixed loop the distance of the leading k invariant directions from the span
# of the whole basis falls by about e^{-(Λ_k - Λ_{k′+1}) τ} per step, with Λ the Floquet
# exponents.
#
# The Galerkin blocks. The same images give the blocks of the Galerkin matrix of the bases before
# the step,
#
#     Eᵢ = Vᵢ₊₁^(r-1)ᵀ ψᵢ Vᵢ^(r-1),   E_m = V₁^(r-1)ᵀ S(s) ψ_m V_m^(r-1),
#
# which the preconditioner uses with these bases, so that E is exact for them, whatever their
# accuracy.
#
# The turn. After the step, the part of the new basis that sticks out of the old span, the new
# basis minus its projection on the old one,
#
#     Zᵢ = Vᵢ^(r) - Vᵢ^(r-1) (Vᵢ^(r-1)ᵀ Vᵢ^(r)),
#
# whose singular values in increasing order, σ₁(Zᵢ) ≤ … ≤ σ_k′(Zᵢ), are the sines of the principal
# angles ϑ_{1,i} ≤ … ≤ ϑ_{k′,i} between the old span at point i and the new. Zᵢ is formed
# explicitly, since the sines computed from the cosines, the singular values of Vᵢ^(r-1)ᵀ Vᵢ^(r),
# would lose the small angles to rounding.
#
# The test. The bases are accurate when, at every point, the k smallest angles are below the
# tolerance ε_sub,
#
#     σ_k(Zᵢ) = sin ϑ_{k,i} ≤ ε_sub,   i = 1, …, m,
#
# whose left side, the largest over the points, is maxturn. Asking for k small angles, and not k′,
# checks only that the span holds the k leading directions, while the other k′ - k, the guard
# vectors, may still be turning: the turn falls at the rate Λ_k - Λ_{k′+1}, with no need of a gap
# after the kth exponent.
#
# The layout. Two indices run over the vectors of the bases, the direction k₁ = 1, …, k′ and the
# shooting point i = 1, …, m: vᵢ^(k₁) is direction k₁ at point i. They are stored by direction. V
# is a vector of k′ elements, each like the seeds, m states, and V[k₁] holds direction k₁ at every
# point,
#
#     V[1] = (v₁¹, v₂¹, …, v_m¹),   …,   V[k′] = (v₁ᵏ′, v₂ᵏ′, …, v_mᵏ′).
#
# The basis of point i is not one object: it is state i of every element, Vᵢ = [vᵢ¹, …, vᵢᵏ′]. With
# m = 3 points and k′ = 2 directions,
#
#                 point 1   point 2   point 3
#     V[1]    (    v₁¹       v₂¹       v₃¹    )     direction 1
#     V[2]    (    v₁²       v₂²       v₃²    )     direction 2
#
# the elements of V are the rows of this table, and the bases its columns, V₁ = [v₁¹, v₁²]. The
# rows are stored, for two reasons:
#
#   - V[k₁] is like the seeds, so that it is shifted and rolled forward as a vector of the Jacobian
#     is, and the propagator takes it as it takes the seeds;
#
#   - an operation between two directions is one operation at all the points. The inner products
#     dots(V[k₁], V[k₂]) are m numbers, one per point, ⟨vᵢ^(k₁), vᵢ^(k₂)⟩, and
#     V[k₂] .-= c .* V[k₁] takes from every point its own multiple. The linear algebra of the m
#     bases, Gram–Schmidt, the Galerkin blocks and the turn at the bottom of the file, is then
#     written once, as loops over the pairs of directions, and runs at all the points together,
#     on a GPU as operations on whole arrays.
#
# A step hands the whole vector V to tangent!, and it is for the propagator to advance the m k′
# states together, with the shooting points integrated once for all the directions. The bases
# before a step and the turn, U and Z, are laid out as V.
#
# Host and device. The device is where the seeds live, a GPU or, for arrays on the CPU, the host
# itself. Two kinds of data, and one passage between them:
#
#   - on the device, what has the size of the states: the elements of V, U and Z, the work seeds
#     tmp, and what dots returns. The inner products of two directions are m numbers, but they
#     come out where the seeds are, in the form that broadcasts against them, and they are used
#     there, as the multipliers of V[k₂] .-= c .* V[k₁]: this never leaves the device;
#
#   - on the host, what has the size k′ × k′ of a point: the triangular factors R and T, the
#     Galerkin blocks E and the sines σ, as k′ × k′ × m and k′ × m arrays of Float64, entry
#     (k₁, k₂, i) for the directions k₁ and k₂ at point i. The small dense algebra, the singular
#     values of T and, in the preconditioner, the coarse matrix and its solves, is done there;
#
#   - from the device to the host go only inner products, the m numbers of a pair of directions
#     at a time, by `vec(Array(c))`, to fill those arrays. On a GPU each is a wait for the device
#     and a small copy. A step makes about 2k′² of them: k′(k′ + 1)/2 for each pass of
#     Gram–Schmidt, two for the bases and one for the turn, and k′² for the Galerkin blocks.
#     Nothing goes from the host to the device.

"""
    Subspace(P, z; k′)

Orthonormal bases of `k′` directions at every shooting point, the seeds of `z`, random at the
start, for subspace iteration around the loop of the propagator `P` at `z`. [`iterate!`](@ref)
makes one subspace step, with the Galerkin blocks and the turn that come with it,
[`maxturn`](@ref) gives the turn of the last step, and `update!(S, z; tol, k)` moves the subspace
to `z` and makes steps there until the turn passes the test.

# Arguments
- `P`: the propagator, with `tangent!`, and `shift!` for relative periodic orbits.
- `z::ShootingVector`: the point, kept, not copied.
- `k′::Int`: the number of directions of each basis, at least 1, required.
"""
mutable struct Subspace{PT, ZT<:ShootingVector, A}
      P::PT                 # the propagator
      z::ZT                 # the point, not a copy: its seeds, period and shifts give the tangent maps
      V::Vector{A}          # device: the bases after the last step, Vᵢ^(r), stored by direction
      U::Vector{A}          # device: the bases before it, Vᵢ^(r-1), which the preconditioner uses
      E::Array{Float64, 3}  # host: their Galerkin blocks Eᵢ = E[:, :, i], k′ × k′ × m, Inf at first
      Z::Vector{A}          # device: the turn of the last step, Zᵢ, then the Q of its QR
      σ::Matrix{Float64}    # host: sin ϑⱼ,ᵢ = σⱼ(Zᵢ), increasing in j, k′ × m, Inf at first
    tmp::A                  # device: the shooting points, then the ends of the segments
  steps::Int                # subspace steps of the last update!

    function Subspace(P, z::ShootingVector; k′::Int)
        m = nsegments(z)

        # ---- input checks ----
        k′ >= 1 ||
            throw(ArgumentError("k′ must be at least 1"))

        # ---- random orthonormal bases: an element like the seeds for every direction ----
        X = z.X
        V = [Random.randn!(similar(X)) for _ in 1:k′]
        _orthonormalise!(V)

        return new{typeof(P), typeof(z), typeof(X)}(P,
                                                    z,
                                                    V,
                                                    similar.(V),
                                                    fill(Inf, k′, k′, m),
                                                    similar.(V),
                                                    fill(Inf, k′, m),
                                                    similar(X),
                                                    0)
    end
end

"""
    iterate!(S::Subspace) -> S

One subspace step at the point of `S`, with the Galerkin blocks and the turn that come with it. The
basis of every shooting point is carried over its segment by the tangent map, that of the last
segment shifted, and the images are orthonormalised at the next point,

    ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r),   S(s) ψ_m V_m^(r-1) = V₁^(r) R_m^(r),

by two passes of modified Gram–Schmidt in the order of the directions. Before Gram–Schmidt the
images give the Galerkin blocks of the bases before the step,
`S.E[:, :, i] = Eᵢ = Vᵢ₊₁^(r-1)ᵀ ψᵢ Vᵢ^(r-1)`.
After it the new bases give the turn Zᵢ and its singular values, `S.σ[:, i]`, the sines of the
principal angles between the old span at point i and the new, in increasing order. The bases before
the step are kept in `S.U`, and the end points of the segments, φ_τ(uᵢ), in `S.tmp`. The k′
directions advance together along the shooting points, by one call of [`tangent!`](@ref) on the
vector `S.V`.

# Arguments
- `S::Subspace`: the subspace, changed.
"""
function iterate!(S::Subspace)
    z  = S.z
    k′ = length(S.V)

    # ---- the length of a segment, the same for all ----
    #      τ = T/m
    τ = period(z)/nsegments(z)

    # ---- the bases before the step, kept in U: the preconditioner uses them with E, and the
    #      turn compares the new ones with them ----
    #      U[k₁] = (v₁^(k₁), …, v_m^(k₁)),   direction k₁ of Vᵢ^(r-1) at every point i
    for k₁ in 1:k′
        copyto!(S.U[k₁], S.V[k₁])
    end

    # ---- the k′ directions advanced together over the segments, along the shooting points: one
    #      call on the vector V as it is. tmp starts as the shooting points and ends as the
    #      end points of the segments, which the preconditioner takes from there ----
    #      V[k₁] = (ψ₁ v₁^(k₁), …, ψ_m v_m^(k₁)),   tmp = (φ_τ(u₁), …, φ_τ(u_m))
    copyto!(S.tmp, z.X)
    tangent!(S.P, S.tmp, S.V, τ)

    # ---- each image at the point it reaches: that of segment i belongs to point i + 1, and that
    #      of the last segment, shifted, to point 1 ----
    #      V[k₁] = (S(s) ψ_m v_m^(k₁), ψ₁ v₁^(k₁), …, ψ_{m-1} v_{m-1}^(k₁))
    for k₁ in 1:k′
        nshifts(z) > 0 && shift!(S.P, tail(S.V[k₁]), shifts(z))
        roll_forward!(S.V[k₁])
    end

    # ---- the Galerkin blocks of the bases before the step: the images, not yet orthonormal, in
    #      the old basis of the point they reach ----
    #      Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E_m = U₁ᵀ S(s) ψ_m U_m
    _galerkin!(S.E, S.U, S.V)

    # ---- the new bases: the images orthonormalised at every point, across the k′ directions ----
    #      ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r)
    _orthonormalise!(S.V)

    # ---- the turn at every point, how far the new span is from the old one, and its singular
    #      values, for the test of the bases ----
    #      Zᵢ = Vᵢ^(r) - Uᵢ (Uᵢᵀ Vᵢ^(r)),   σⱼ(Zᵢ) = sin ϑⱼ,ᵢ
    _turn!(S.Z, S.σ, S.U, S.V)

    return S
end

"""
    maxturn(S::Subspace; k=k′) -> Float64

The turn of the last subspace step: the sine of the `k`-th smallest principal angle between the
span of the bases before the step and the span after it, the largest over the shooting points,

    maxᵢ σ_k(Zᵢ) = maxᵢ sin ϑ_{k,i},

the left side of the test of the bases, σ_k(Zᵢ) ≤ ε_sub at every point. Once the span holds `k`
directions that the tangent maps carry into it, the `k` smallest angles vanish, whatever the other
directions, the guard vectors, do. Inf before the first step.

# Arguments
- `S::Subspace`: the subspace.
- `k::Int=k′`: the number of leading directions that the test checks, from 1 to the number `k′` of
  directions of each basis.

The sines are the singular values of Zᵢ, formed by [`iterate!`](@ref), from the triangular factor
of its QR factorisation by modified Gram–Schmidt: they are resolved down to rounding, eps times the
largest.
"""
function maxturn(S::Subspace; k::Int=length(S.V))
    k′ = length(S.V)

    # ---- input checks ----
    1 <= k <= k′ ||
        throw(ArgumentError("k must be between 1 and k′"))

    # ---- the kth smallest sine at every point, the largest over the points ----
    #      maxᵢ σ_k(Zᵢ)
    return maximum(S.σ[k, :])
end

"""
    update!(S::Subspace, z; tol, k=k′, maxsteps=100, cap=:error) -> S

Move `S` to the point `z`, with no integration, and make subspace steps there from the bases it
has, at least one, until the bases pass the test of the paper, the turn of the leading `k`
directions below `tol` at every shooting point,

    maxturn(S; k) = maxᵢ σ_k(Zᵢ) ≤ ε_sub,   ε_sub = tol,

or `maxsteps` steps have been made. A function of this package, not the `update!` of NKRoots.jl.
After a move the first step measures the lag, how far the span turns at the new point. A positive
`tol` keeps the bases up to a tolerance, and `tol = 0` with `cap = :keep` makes a fixed number of
steps, `maxsteps`, two of the ways of [`Preconditioner`](@ref).

# Arguments
- `S::Subspace`: the subspace, changed; `S.steps` holds the number of steps made.
- `z::ShootingVector`: the new point, kept, not copied.
- `tol::Real`: the tolerance ε_sub on the sine of the turn, as [`maxturn`](@ref).
- `k::Int=k′`: the number of leading directions that the test checks, from 1 to the number `k′`
  of directions of each basis.
- `maxsteps::Int=100`: the most steps.
- `cap::Symbol=:error`: when the bases fail the test after `maxsteps` steps, `:error` throws an
  error and `:keep` keeps them as they are. With `tol = 0` and `:keep` the update makes exactly
  `maxsteps` steps.
"""
function update!(S::Subspace, z::ShootingVector; tol::Real,
                                                 k::Int=length(S.V),
                                                 maxsteps::Int=100,
                                                 cap::Symbol=:error)
    # ---- input checks ----
    1 <= k <= length(S.V) ||
        throw(ArgumentError("k must be between 1 and k′"))
    maxsteps >= 1 ||
        throw(ArgumentError("maxsteps must be at least 1"))
    cap in (:error, :keep) ||
        throw(ArgumentError("cap must be :error or :keep"))

    # ---- the tolerance by its name in the paper ----
    ε_sub = tol

    # ---- the subspace moved to z, with no integration ----
    S.z = z

    # ---- steps until the turn of the leading k directions is below ε_sub at every point ----
    #      σ_k(Zᵢ) ≤ ε_sub,   i = 1, …, m
    S.steps = 0
    while S.steps < maxsteps
        iterate!(S)
        S.steps += 1
        maxturn(S; k) <= ε_sub && return S
    end

    # ---- the test failed after the cap: the bases kept as they are, or an error ----
    cap === :keep &&
        return S
    error("the bases failed the test in $(maxsteps) subspace steps: the turn of the $(k) leading " *
          "directions is $(maxturn(S; k)), above $(ε_sub)")
end

# ---------------------------------------------------------------------------- #
# the algebra of the bases                                                     #
# ---------------------------------------------------------------------------- #

# The four functions below work on bases stored by direction (see the layout at the top): U, V and
# Z are vectors of k′ elements like the seeds, element k₁ the direction k₁ at every point. Each is a
# loop over the pairs of directions, and every line of the loop acts at the m points at once:
#
#     c = dots(U[k₁], V[k₂])      the m inner products ⟨uᵢ^(k₁), vᵢ^(k₂)⟩, one per point
#     V[k₂] .-= c .* U[k₁]        at every point i, vᵢ^(k₂) ← vᵢ^(k₂) - cᵢ uᵢ^(k₁)
#
# The inner product is that of the seeds, dots, with its weights if it has them. U, V, Z and c
# are on the device, and so are both lines above. Only the coefficients go to the host, m numbers
# a pair, into the k′ × k′ × m arrays of the small matrices, which are of the host, by
# `vec(Array(c))`. dots gives its m numbers in the form that broadcasts against the seeds, and
# where they live: for an array, one with every dimension of size one but the last, 1 × m or
# 1 × 1 × m, on a GPU if the seeds are. `Array` copies them to the host, and `vec` makes of them
# a plain vector of m numbers, as an index over the points needs.

# The Galerkin blocks of the bases U before a step, from their images V, taken before
# Gram–Schmidt. The step has rolled the images forward, so that V[k₂] holds at point j the image
# that reaches it, that of segment j - 1, cyclically,
#
#     V[k₂] = (S(s) ψ_m u_m^(k₂), ψ₁ u₁^(k₂), …, ψ_{m-1} u_{m-1}^(k₂)).
#
# Its coordinates in the old basis of the point it reaches are the block of the segment it comes
# from,
#
#     Eᵢ[k₁, k₂] = ⟨uᵢ₊₁^(k₁), ψᵢ uᵢ^(k₂)⟩,   Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E_m = U₁ᵀ S(s) ψ_m U_m.
#
# For a pair of directions dots(U[k₁], V[k₂]) gives them at all the points, in the order of the
# point reached, j = i + 1: moved back by one place they are in the order of the segment, i. E is
# k′ × k′ × m on the host, E[:, :, i] the block of segment i. U and V are on the device and are
# not changed: every pair of directions costs one reduction there and one copy to the host, k′²
# in all.
function _galerkin!(E::Array{Float64, 3}, U::Vector, V::Vector)
    k′ = length(U)

    for k₁ in 1:k′, k₂ in 1:k′
        # ---- at every point j, the coordinate along uⱼ^(k₁) of the image that reaches it, on the
        #      host as a vector of m numbers: without vec the shift below would run along a
        #      dimension of size one, and move nothing ----
        #      cⱼ = ⟨uⱼ^(k₁), ψⱼ₋₁ uⱼ₋₁^(k₂)⟩,   j = 1, …, m
        c = vec(Array(dots(U[k₁], V[k₂])))

        # ---- moved back to the segment the image comes from ----
        #      Eᵢ[k₁, k₂] = cᵢ₊₁ = ⟨uᵢ₊₁^(k₁), ψᵢ uᵢ^(k₂)⟩
        E[k₁, k₂, :] .= circshift(c, -1)
    end

    return E
end

# The turn of a step at every point, from the bases U before it and V after it: how far the new
# span is from the old. In three stages, each at all the points at once:
#
#     the coordinates of the new basis in the old,      Cᵢ = Uᵢᵀ Vᵢ
#     the part of the new basis outside the old span,    Zᵢ = Vᵢ - Uᵢ Cᵢ,   formed in Z
#     its singular values, increasing, into σ[:, i],     σⱼ(Zᵢ) = sin ϑⱼ,ᵢ
#
# C is a k′ × k′ matrix whose entry (k₁, k₂) is not a number but the m of them that dots gives,
# one per point, left in that form and on the device, since each only multiplies U[k₁] there,
# point by point: the coordinates never come to the host. Z is stored as V is, on the device,
# Z[k₂] the direction k₂ of the turn at every point. What comes to the host is T alone, through
# Gram–Schmidt, and σ is computed there.
#
# The singular values of Zᵢ are not computed on Zᵢ itself, for three reasons. Zᵢ is n × k′, with n
# the size of a state, a whole field, possibly on a GPU. It is not one matrix: its columns are
# state i of each of the k′ elements of Z. And its singular values are meant in the inner product
# of the seeds, dots, which may have weights, where a numerical SVD takes the Euclidean one. So Zᵢ
# is factorised first, in that inner product,
#
#     Zᵢ = Qᵢ Tᵢ,   Qᵢ with orthonormal columns in dots,   Tᵢ upper triangular, k′ × k′,
#
# by one pass of modified Gram–Schmidt, at all the points at once, which leaves Qᵢ in Z and
# returns T, k′ × k′ × m on the host. Qᵢ being orthonormal, Zᵢ and Tᵢ have the same singular
# values,
#
#     σⱼ(Zᵢ) = σⱼ(Tᵢ),
#
# and those of the small matrix T[:, :, i] are computed on the host, point by point, at no cost.
#
# What this takes for granted:
#
#   - one pass is enough. It leaves Qᵢ orthonormal only to about eps times the condition number
#     of Zᵢ, which is large when some directions have settled and others have not, but the
#     triangular factor is backward stable however far Qᵢ is from orthonormal, and only the
#     factor is used;
#
#   - the small sines survive. The factor resolves them down to eps times the largest. The other
#     cheap way, the eigenvalues of the k′ × k′ matrix ZᵢᵀZᵢ, from dots alone, would square them
#     and lose them below √eps times the largest, about 1e-8, above the tolerances of the test;
#
#   - a direction that does not turn at a point gives a zero column of Zᵢ there, which
#     Gram–Schmidt leaves at zero, with a zero on the diagonal of Tᵢ: a zero sine, and no
#     division by zero.
#
# The sines are capped at 1, against rounding.
function _turn!(Z::Vector, σ::Matrix{Float64}, U::Vector, V::Vector)
    k′ = length(V)
    m  = size(σ, 2)

    # ---- the coordinates of the new basis in the old: for every pair of directions, m numbers,
    #      one per point ----
    #      Cᵢ[k₁, k₂] = ⟨uᵢ^(k₁), vᵢ^(k₂)⟩
    C = [dots(U[k₁], V[k₂]) for k₁ in 1:k′, k₂ in 1:k′]

    # ---- the new basis minus its projection on the old, direction by direction ----
    #      zᵢ^(k₂) = vᵢ^(k₂) - Σ_k₁ Cᵢ[k₁, k₂] uᵢ^(k₁)
    for k₂ in 1:k′
        copyto!(Z[k₂], V[k₂])
        for k₁ in 1:k′
            Z[k₂] .-= C[k₁, k₂] .* U[k₁]
        end
    end

    # ---- its singular values, those of its triangular factor, point by point on the host, in
    #      increasing order ----
    #      Zᵢ = Qᵢ Tᵢ,   σⱼ(Zᵢ) = σⱼ(Tᵢ) = sin ϑⱼ,ᵢ
    T = _gram_schmidt!(Z)
    for i in 1:m
        σ[:, i] .= sort(min.(LinearAlgebra.svdvals(T[:, :, i]), 1.0))
    end

    return σ
end

# The QR factorisation of a step, at every point: the images V, stored by direction, overwritten
# with the new orthonormal bases,
#
#     ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r),
#
# by two passes of modified Gram–Schmidt. One pass leaves the vectors orthogonal only to about eps
# times the condition number of the images, which the tangent maps make large, as they stretch
# the leading directions and shrink the others. The second, on vectors that are nearly orthonormal
# already, brings them to rounding. The triangular factor Rᵢ^(r), the product of those of the two
# passes, is not needed, and not kept.
function _orthonormalise!(V::Vector)
    for _ in 1:2
        _gram_schmidt!(V)
    end

    return V
end

# One pass of modified Gram–Schmidt on the bases V, stored by direction, at all the points at
# once. V is overwritten with Q, and the triangular factors are returned,
#
#     Vᵢ = Qᵢ Rᵢ,   i = 1, …, m,
#
# in R, k′ × k′ × m on the host: R[:, :, i] is upper triangular, with a positive diagonal. V is on
# the device and stays there: a pass makes k′(k′ + 1)/2 copies to the host, the products of every
# pair of directions and the norms, m numbers each.
#
# The loop is that of the textbook on the columns of one matrix, with the element V[k₂] for column
# k₂. The directions are taken in order: each loses its components along those before it, which are
# orthonormal by then, and is normalised, so that the first j vectors of Qᵢ span the first j of
# Vᵢ. The scheme is the modified one: the component along q^(k₁) is measured on the vector as it
# stands, the components before already taken out, and not on the vector at the start.
#
# What differs from the textbook is that a line acts at the m points together, since V[k₂] holds
# direction k₂ at all of them. c and n are not numbers but m of them, one per point, the inner
# products and the norms there, in the form that dots gives, which broadcasts against V[k₂] so that
# every point gets its own. They go to the host only to fill R. The inner product is that of the
# seeds, so that the vectors come out orthonormal in the geometry of the states.
#
# A vector that vanishes at a point stays zero there, with a zero on the diagonal of R: the
# division by its norm is guarded.
function _gram_schmidt!(V::Vector)
    k′ = length(V)
    m  = nsegments(first(V))
    R  = zeros(k′, k′, m)

    for k₂ in 1:k′
        # ---- direction k₂ minus its components along the directions before it, one after the
        #      other, at every point. c stays where the seeds are for the subtraction, and goes to
        #      the host, as a vector over the points, for R ----
        #      cᵢ = ⟨qᵢ^(k₁), vᵢ^(k₂)⟩,   vᵢ^(k₂) ← vᵢ^(k₂) - cᵢ qᵢ^(k₁),   Rᵢ[k₁, k₂] = cᵢ,
        #      for k₁ = 1, …, k₂ - 1
        for k₁ in 1:k₂-1
            c = dots(V[k₁], V[k₂])
            V[k₂] .-= c .* V[k₁]
            R[k₁, k₂, :] .= vec(Array(c))
        end

        # ---- direction k₂ normalised at every point, and the norms to the host likewise ----
        #      nᵢ = ‖vᵢ^(k₂)‖,   qᵢ^(k₂) = vᵢ^(k₂)/nᵢ,   Rᵢ[k₂, k₂] = nᵢ
        n = sqrt.(dots(V[k₂], V[k₂]))
        V[k₂] ./= max.(n, floatmin(Float64))
        R[k₂, k₂, :] .= vec(Array(n))
    end

    return R
end
