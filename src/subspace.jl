# Subspace iteration around the loop: the bases of the leading directions of the tangent maps at
# every shooting point, which the preconditioner of preconditioner.jl is built on. The iteration is
# the subspace (orthogonal) iteration of Saad (2011), as in the Newton–Picard method of Lust (1997)
# and Lust et al. (1998); the step differs from Lust's sweep, since every seed carries its own basis
# over its own segment, all at once. The test of the bases, by the turn of the span, and the three
# ways of keeping them up (frozen, fixed steps, to a tolerance; see preconditioner.jl) are from
# Lasagna et al. (2026).
#
# Names. Seeds u₁, …, u_m, segments of length τ = T/m, ψᵢ = Dφ_τ(uᵢ) the tangent map of segment i,
# Λ₁ ≥ Λ₂ ≥ … the Floquet exponents. Every seed carries a basis of k′ directions, of which the
# test checks the leading k ≤ k′: the keywords `k′` and `k`.
#
# The step. Vᵢ is the orthonormal basis at point i, r the step. Step r carries the basis of every
# point over its segment by the tangent map and orthonormalises the images at the next point,
#
#     ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r),   i = 1, …, m - 1,   S(s) ψ_m V_m^(r-1) = V₁^(r) R_m^(r),
#
# a QR factorisation by two passes of modified Gram–Schmidt, Rᵢ^(r) upper triangular. The k′
# directions of the m segments go to the tangent L in one call, which integrates the tangent
# equation along the seeds. On a fixed loop the distance of the leading k invariant directions from
# the span of the basis falls by about e^{-(Λ_k - Λ_{k′+1}) τ} per step (Saad 2011, Theorem 5.2).
#
# The Galerkin blocks. The same images give the blocks of the Galerkin matrix of the bases before
# the step, U = V^(r-1),
#
#     Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E_m = U₁ᵀ S(s) ψ_m U_m,
#
# which the preconditioner uses with these bases: E is exact for them, whatever their accuracy.
#
# The turn. After the step, the part of the new basis outside the old span,
#
#     Zᵢ = Vᵢ^(r) - Uᵢ (Uᵢᵀ Vᵢ^(r)),
#
# whose singular values in increasing order, σ₁(Zᵢ) ≤ … ≤ σ_k′(Zᵢ), are the sines of the principal
# angles ϑ₁,ᵢ ≤ … ≤ ϑ_k′,ᵢ between the old span at point i and the new. Zᵢ is formed explicitly: the
# sines from the cosines, the singular values of UᵢᵀVᵢ, would lose the small angles to rounding.
#
# The test (Lasagna et al. 2026). The bases are accurate when, at every point, the k smallest
# angles are below the tolerance ε_sub,
#
#     σ_k(Zᵢ) = sin ϑ_{k,i} ≤ ε_sub,   i = 1, …, m,
#
# whose left side, the largest over the points, is `maxturn`. It asks for k small angles, not k′: it
# checks that the span holds the k leading directions, while the k′ - k guard vectors may still be
# turning. The two rates differ. The turn falls by about e^{-(Λ_k - Λ_{k+1}) τ} per step when k′ = k,
# and faster, by about e^{-(Λ_k - Λ_{2k′-k+1}) τ}, when k′ > k, as observed on every pair of k and k′
# tested (cf. Zhou et al. 2022, Theorem 3.5), with no need of a gap after Λ_k. The distance of the
# leading k directions from the span falls only at Λ_k - Λ_{k′+1}, as above, so a turn below ε_sub
# does not put the directions within ε_sub of the span.
#
# The layout. The bases are stored by direction. U, V, Z are vectors of k′ elements, each like the
# seeds (m states), and V[k₁] holds direction k₁ at every point, vᵢ^(k₁) at point i. With m = 3
# and k′ = 2,
#
#                 point 1   point 2   point 3
#     V[1]    (    v₁¹       v₂¹       v₃¹    )     direction 1
#     V[2]    (    v₁²       v₂²       v₃²    )     direction 2
#
# the elements of V are the rows of this table and the basis Vᵢ = [vᵢ¹, vᵢ²] is a column. Rows are
# stored for two reasons:
#
#   - V[k₁] is like the seeds, so that the tangent takes it as it takes them, and it is shifted
#     and rolled forward as a vector of the Jacobian is;
#
#   - an operation between two directions is one operation at all the points: dots(V[k₁], V[k₂])
#     gives the m inner products, one per point, and V[k₂] .-= c .* V[k₁] takes from every point
#     its own multiple. Gram–Schmidt, the Galerkin blocks and the turn are then loops over the
#     pairs of directions, and on a GPU run as operations on whole arrays.
#
# Host and device. The device is where the seeds live, a GPU or, for arrays on the CPU, the host
# itself.
#
#   - on the device, what has the size of the states: the elements of V, U and Z, the work seeds
#     tmp, and the m inner products that dots returns, in the form that broadcasts against the
#     seeds. They are used there, as multipliers, and never leave;
#
#   - on the host, what has the size k′ × k′ of a point: the triangular factors R and T, the
#     Galerkin blocks E (k′ × k′ × m, entry (k₁, k₂, i)) and the sines σ (k′ × m), all Float64. The
#     small dense algebra, the singular values of T and, in the preconditioner, the coarse
#     matrix and its solves, is done there;
#
#   - from the device to the host go only inner products, m numbers a pair of directions, by
#     `vec(Array(c))`; on a GPU each is a wait and a small copy. A step makes about 5k′²/2 of
#     them: k′(k′ + 1)/2 for each of the three Gram–Schmidt passes (two for the bases, one for
#     the turn) and k′² for the Galerkin blocks. Nothing goes from the host to the device.

"""
    Subspace(L, z; k′, S=nothing)

Orthonormal bases of `k′` directions at every shooting point, the seeds of `z`, random at the
start, for subspace iteration around the loop with the tangent `L`. [`iterate!`](@ref) makes one
step, with the Galerkin blocks and the turn that come with it; [`maxturn`](@ref) gives the turn of
the last step; `update!(S, z; tol, k)` moves the subspace to `z` and steps until the bases pass the
test of Lasagna et al. (2026).

    step:    ψᵢ Vᵢ = Vᵢ₊₁′ Rᵢ,   S(s) ψ_m V_m = V₁′ R_m      (V′ the bases after the step)
    turn:    Zᵢ = Vᵢ′ - Vᵢ (Vᵢᵀ Vᵢ′),   σⱼ(Zᵢ) = sin ϑⱼ,ᵢ

The bases are stored by direction: `S.V` is a vector of `k′` elements like the seeds, `S.V[k₁]`
holding direction `k₁` at every point.

# Arguments
- `L`: the tangent, called as `L(X, V, τ)` with a vector `V` of `k′` perturbations of every seed.
- `z::ShootingVector`: the point, kept, not copied.
- `k′::Int`: the number of directions of each basis, at least 1, required.
- `S=nothing`: the shift, called as `S(X, s...)`, for relative periodic orbits.
"""
mutable struct Subspace{LT, ST, ZT<:ShootingVector, A}
           L::LT                   # the tangent
           S::ST                   # the shift, or nothing for periodic orbits
           z::ZT                   # the point, not a copy: its seeds, period and shifts give
                                   # the tangent maps
           V::Vector{A}            # device: the bases after the last step, Vᵢ^(r), stored by
                                   # direction
           U::Vector{A}            # device: the bases before it, Vᵢ^(r-1), which the
                                   # preconditioner uses
           E::Array{Float64, 3}    # host: their Galerkin blocks Eᵢ = E[:, :, i], k′ × k′ × m,
                                   # Inf at first
           Z::Vector{A}            # device: the turn of the last step, Zᵢ, then the Q of its QR
           σ::Matrix{Float64}      # host: sin ϑⱼ,ᵢ = σⱼ(Zᵢ), increasing in j, k′ × m, Inf at first
         tmp::A                    # device: the shooting points, then the ends of the segments
       steps::Int                  # subspace steps of the last update!

    function Subspace(L, z::ShootingVector; k′::Int, S=nothing)
        m = nsegments(z)

        # ---- input checks ----
        k′ >= 1 ||
            throw(ArgumentError("k′ must be at least 1"))
        nshifts(z) > 0 && S === nothing &&
            throw(ArgumentError("a relative periodic orbit needs its shift S"))

        # ---- random orthonormal bases: an element like the seeds for every direction ----
        X = z.X
        V = [Random.randn!(similar(X)) for _ in 1:k′]
        _orthonormalise!(V)

        return new{typeof(L),
                   typeof(S),
                   typeof(z),
                   typeof(X)}(L,
                              S,
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
segment shifted, and the images are orthonormalised at the next point, by two passes of modified
Gram–Schmidt in the order of the directions,

    ψᵢ Vᵢ = Vᵢ₊₁′ Rᵢ,   S(s) ψ_m V_m = V₁′ R_m      (V before the step, V′ after)

The images, before Gram–Schmidt, give the Galerkin blocks of the bases before the step,

    Eᵢ = Vᵢ₊₁ᵀ ψᵢ Vᵢ,   E_m = V₁ᵀ S(s) ψ_m V_m,       S.E[:, :, i] = Eᵢ

and the new bases give the turn,

    Zᵢ = Vᵢ′ - Vᵢ (Vᵢᵀ Vᵢ′),   S.σ[:, i] = (σ₁(Zᵢ), …, σ_k′(Zᵢ)) = sines of the principal angles
                                            between the spans of Vᵢ and Vᵢ′, increasing

The k′ directions advance together along the seeds, by one call of the tangent on `S.V`.

After the step `S.V` holds the new bases, `S.U` the bases before it, which go with `S.E`, and
`S.tmp` the end points of the segments, φ_τ(uᵢ), unshifted.

# Arguments
- `S::Subspace`: the subspace, changed.
"""
function iterate!(S::Subspace)
    z  = S.z
    k′ = length(S.V)

    # ---- the length of a segment, the same for all, or of each segment ----
    #      τ = T/m,   or τᵢ = wᵢ T
    τ = times(z)

    # ---- the bases before the step, kept in U: the preconditioner uses them with E, and the
    #      turn compares the new ones with them ----
    #      U[k₁] = (u₁^(k₁), …, u_m^(k₁)),   direction k₁ of Uᵢ = Vᵢ^(r-1) at every point i
    for k₁ in 1:k′
        copyto!(S.U[k₁], S.V[k₁])
    end

    # ---- the k′ directions advanced together over the segments, in one call on the vector V.
    #      tmp starts as the seeds and ends as the end points of the segments, which the
    #      preconditioner takes from there ----
    #      V[k₁] = (ψ₁ u₁^(k₁), …, ψ_m u_m^(k₁)),   tmp = (φ_τ(u₁), …, φ_τ(u_m))
    copyto!(S.tmp, z.X)
    S.L(S.tmp, S.V, τ)

    # ---- each image at the point it reaches: that of segment i belongs to point i + 1, and that
    #      of the last segment, shifted, to point 1 ----
    #      V[k₁] = (S(s) ψ_m u_m^(k₁), ψ₁ u₁^(k₁), …, ψ_{m-1} u_{m-1}^(k₁))
    for k₁ in 1:k′
        nshifts(z) > 0 && S.S(tail(S.V[k₁]), shifts(z)...)
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
span of the bases before the step and that after it, the largest over the shooting points,

    maxᵢ σ_k(Zᵢ) = maxᵢ sin ϑ_{k,i}

the left side of the test of Lasagna et al. (2026), σ_k(Zᵢ) ≤ ε_sub at every point. It vanishes
when the span holds `k` directions that the tangent maps carry into it, whatever the other
directions, the guard vectors, do. `Inf` before the first step.

The sines are the singular values of the triangular factor of Zᵢ, by modified Gram–Schmidt: they
are resolved down to rounding, eps times the largest.

# Arguments
- `S::Subspace`: the subspace.
- `k::Int=k′`: the number of leading directions that the test checks, from 1 to `k′`.
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

Move `S` to the point `z`, with no integration, and step there from the bases it has, at least
once, until the test of Lasagna et al. (2026) is passed,

    maxturn(S; k) = maxᵢ σ_k(Zᵢ) ≤ ε_sub = tol

or `maxsteps` steps have been made. The first step after a move measures the lag, how far the span
turns at the new point. Not exported, and not the `update!` of NKRoots.jl, which the preconditioner
extends. The keywords give two of the three ways of keeping the bases up: `tol > 0`, to a tolerance;
`tol = 0` with `cap = :keep`, a fixed number `maxsteps` of steps. The third, frozen, is `every = 0`
of [`Preconditioner`](@ref).

# Arguments
- `S::Subspace`: the subspace, changed; `S.steps` holds the number of steps made.
- `z::ShootingVector`: the new point, kept, not copied.
- `tol::Real`: the tolerance ε_sub on the sine of the turn, as [`maxturn`](@ref).
- `k::Int=k′`: the number of leading directions that the test checks, from 1 to `k′`.
- `maxsteps::Int=100`: the most steps.
- `cap::Symbol=:error`: when the test fails after `maxsteps` steps, `:error` throws an error and
  `:keep` keeps the bases as they are.
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

    # ---- the tolerance by its name in the test ----
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

# The four functions below work on bases stored by direction (see the layout at the top). Each is
# a loop over the pairs of directions, and every line of it acts at the m points at once:
#
#     c = dots(U[k₁], V[k₂])      the m inner products ⟨uᵢ^(k₁), vᵢ^(k₂)⟩, one per point
#     V[k₂] .-= c .* U[k₁]        at every point i, vᵢ^(k₂) ← vᵢ^(k₂) - cᵢ uᵢ^(k₁)
#
# The inner product is that of the seeds, dots, with its weights if it has them. Both lines run on
# the device; only the coefficients go to the host, by `vec(Array(c))`, to fill the arrays of the
# small matrices. dots gives its m numbers in the form that broadcasts against the seeds, with
# every dimension of size one but the last (1 × m or 1 × 1 × m for arrays): `Array` copies them
# to the host and `vec` makes of them a plain vector of m numbers, as an index over the points
# needs.

# The Galerkin blocks of the bases U before a step, from their images V, taken before Gram–Schmidt.
# The step has rolled the images forward, so that V[k₂] holds at point j the image that reaches it,
# that of segment j - 1, cyclically,
#
#     V[k₂] = (S(s) ψ_m u_m^(k₂), ψ₁ u₁^(k₂), …, ψ_{m-1} u_{m-1}^(k₂)).
#
# Its coordinates in the old basis of the point it reaches are the block of the segment it comes
# from,
#
#     Eᵢ[k₁, k₂] = ⟨uᵢ₊₁^(k₁), ψᵢ uᵢ^(k₂)⟩,   Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E_m = U₁ᵀ S(s) ψ_m U_m.
#
# dots(U[k₁], V[k₂]) gives them in the order of the point reached, j = i + 1; moved back by one
# place they are in the order of the segment. E is k′ × k′ × m on the host, E[:, :, i] the block of
# segment i. U and V are not changed; a pair of directions costs one reduction and one copy to the
# host, k′² in all.
function _galerkin!(E::Array{Float64, 3}, U::Vector, V::Vector)
    k′ = length(U)

    for k₁ in 1:k′, k₂ in 1:k′
        # ---- at every point j, the coordinate along uⱼ^(k₁) of the image that reaches it, on the
        #      host as a vector of m numbers: without vec, circshift would act on a dimension of
        #      size one and move nothing ----
        #      cⱼ = ⟨uⱼ^(k₁), ψⱼ₋₁ uⱼ₋₁^(k₂)⟩,   j = 1, …, m
        c = vec(Array(dots(U[k₁], V[k₂])))

        # ---- moved back to the segment the image comes from ----
        #      Eᵢ[k₁, k₂] = cᵢ₊₁ = ⟨uᵢ₊₁^(k₁), ψᵢ uᵢ^(k₂)⟩
        E[k₁, k₂, :] .= circshift(c, -1)
    end

    return E
end

# The turn of a step at every point, from the bases U before it and V after it. In three stages,
# each at all the points at once:
#
#     the coordinates of the new basis in the old,      Cᵢ = Uᵢᵀ Vᵢ
#     the part of the new basis outside the old span,    Zᵢ = Vᵢ - Uᵢ Cᵢ,   formed in Z
#     its singular values, increasing, into σ[:, i],     σⱼ(Zᵢ) = sin ϑⱼ,ᵢ
#
# Cᵢ[k₁, k₂] is, for every pair of directions, the m numbers of dots, left on the device in that
# form since each only multiplies U[k₁] there: the coordinates never come to the host. Z is stored
# as V is. Only T, below, comes to the host.
#
# The singular values are not taken of Zᵢ itself: Zᵢ is n × k′, with n the size of a state, a
# whole field, possibly on a GPU; it is not one matrix, its columns being state i of each element
# of Z; and its singular values are meant in the inner product of the seeds, which may have
# weights, where an SVD takes the Euclidean one. So Zᵢ is first factorised in that inner product,
#
#     Zᵢ = Qᵢ Tᵢ,   Qᵢ with orthonormal columns in dots,   Tᵢ upper triangular, k′ × k′,
#
# by one pass of modified Gram–Schmidt at all the points, which leaves Qᵢ in Z and returns T,
# k′ × k′ × m on the host. Qᵢ being orthonormal, σⱼ(Zᵢ) = σⱼ(Tᵢ), and the SVD of the small
# T[:, :, i] is done on the host, point by point, at no cost.
#
# Why this is enough:
#
#   - one pass leaves Qᵢ orthonormal only to about eps times the condition number of Zᵢ, large
#     when some directions have settled and others have not; but the triangular factor is backward
#     stable however far Qᵢ is from orthonormal, and only it is used;
#
#   - the small sines survive: T resolves them down to eps times the largest. The eigenvalues of
#     ZᵢᵀZᵢ, from dots alone, would square them and lose those below √eps times the largest, about
#     1e-8, above the tolerances of the test;
#
#   - a direction that does not turn gives a zero column of Zᵢ, which Gram–Schmidt leaves at zero
#     with a zero on the diagonal of Tᵢ: a zero sine, and no division by zero.
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

# The QR factorisation of a step at every point: the images V, stored by direction, overwritten
# with the new orthonormal bases,
#
#     ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r),
#
# by two passes of modified Gram–Schmidt. One pass leaves the vectors orthogonal only to about eps
# times the condition number of the images, which the tangent maps make large by stretching the
# leading directions and shrinking the others; the second, on vectors nearly orthonormal already,
# brings them to rounding. The factor Rᵢ^(r), the product of those of the two passes, is not needed.
function _orthonormalise!(V::Vector)
    for _ in 1:2
        _gram_schmidt!(V)
    end

    return V
end

# One pass of modified Gram–Schmidt on the bases V, stored by direction, at all the points at once.
# V is overwritten with Q and the triangular factors are returned,
#
#     Vᵢ = Qᵢ Rᵢ,   i = 1, …, m,
#
# in R, k′ × k′ × m on the host: R[:, :, i] is upper triangular with a nonnegative diagonal. A
# pass makes k′(k′ + 1)/2 copies to the host, the products of every pair of directions and the
# norms, m numbers each.
#
# It is the textbook loop on the columns of one matrix, V[k₂] being column k₂: each direction loses
# its components along those before it, orthonormal by then, and is normalised, so that the first
# j vectors of Qᵢ span the first j of Vᵢ. It is the modified scheme: the component along q^(k₁) is
# measured on the vector as it stands, with the earlier components already taken out.
#
# A line acts at the m points together: c and n are not numbers but m of them, one per point, in
# the form of dots, which broadcasts against V[k₂] so that every point gets its own. The inner
# product is that of the seeds, so that the vectors come out orthonormal in the geometry of the
# states. A vector that vanishes at a point stays zero there, with a zero on the diagonal of R:
# the division by its norm is guarded.
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
