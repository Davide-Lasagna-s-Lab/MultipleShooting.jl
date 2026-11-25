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
# directions of the m segments form one batch of m k′ states, which tangent! advances in one call
# along the shooting points, so that a step costs k′ integrations of the tangent equation over the
# orbit, and one of the equation when the propagator shares it among the directions. On a fixed loop the distance of the leading k
# invariant directions from the span of the whole basis falls by about e^{-(Λ_k - Λ_{k′+1}) τ} per
# step, with Λ the Floquet exponents.
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
# The layout. The bases are k′ batches like the shooting points, one per direction: V[a] holds
# direction a at every point,
#
#     V[1] = (v₁¹, v₂¹, …, v_m¹),   …,   V[k′] = (v₁ᵏ′, v₂ᵏ′, …, v_mᵏ′),
#
# so that each is shifted and moved forward as a vector of the Jacobian is. A step packs them into one
# batch of k′ blocks of m states,
#
#     V = (v₁¹, …, v_m¹, v₁², …, v_m², …),
#
# which tangent! advances along the shooting points, so that a propagator on a GPU advances the
# m k′ states together, and may integrate the shooting points once for all the directions.
# Gram–Schmidt runs across the k′ batches, state by state, in the order of the directions.

"""
    Subspace(P, z; k′)

Orthonormal bases of `k′` directions at every shooting point, the seeds of `z`, random at the
start, for subspace iteration around the loop of the propagator `P` at `z`. [`iterate!`](@ref)
makes one subspace step, with the Galerkin blocks and the turn that come with it,
[`maxturn`](@ref) gives the turn of the last step, and `update!(S, z; tol, k)` moves the subspace
to `z` and makes steps there until the turn passes the test.

# Arguments
- `P`: the propagator, with `tangent!`, and `shift!` for relative periodic orbits.
- `z::MultiVector`: the point, kept, not copied.
- `k′::Int`: the number of directions of each basis, at least 1, required.
"""
mutable struct Subspace{PT, ZT<:MultiVector, A, B}
      P::PT                 # the propagator
      z::ZT                 # the point, not a copy: its seeds, period and shifts give the tangent maps
      V::Vector{A}          # the bases after the last step, Vᵢ^(r): V[a] holds direction a
      U::Vector{A}          # the bases before it, Vᵢ^(r-1), which the preconditioner uses with E
      E::Array{Float64, 3}  # their Galerkin blocks Eᵢ = E[:, :, i], k′ × k′ × m, Inf before any step
      Z::Vector{A}          # the turn of the last step, Zᵢ, then the orthonormal factor of its QR
      σ::Matrix{Float64}    # sin ϑⱼ,ᵢ = σⱼ(Zᵢ), increasing in j, k′ × m, Inf before any step
    tmp::A                  # work batch: the shooting points, then the end points of the segments
    Vk′::B                  # the packed batch of the bases, k′ blocks of m states, block a direction a
  steps::Int                # subspace steps of the last update!

    function Subspace(P, z::MultiVector; k′::Int)
        m = nsegments(z)

        # ---- input checks ----
        k′ >= 1 ||
            throw(ArgumentError("k′ must be at least 1"))

        # ---- random orthonormal bases ----
        X = z.X
        V = [Random.randn!(similar(X)) for _ in 1:k′]
        _orthonormalise!(V)

        # ---- the packed batch of a step, m k′ states ----
        Vk′ = similar_states(X, k′*m)

        return new{typeof(P), typeof(z), typeof(X), typeof(Vk′)}(P,
                                                                 z,
                                                                 V,
                                                                 similar.(V),
                                                                 fill(Inf, k′, k′, m),
                                                                 similar.(V),
                                                                 fill(Inf, k′, m),
                                                                 similar(X),
                                                                 Vk′,
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
directions advance together, as one batch of m k′ states along the shooting points, by one call of
[`tangent!`](@ref).

# Arguments
- `S::Subspace`: the subspace, changed.
"""
function iterate!(S::Subspace)
    z  = S.z
    k′ = length(S.V)
    m  = nsegments(z)

    # ---- the bases before the step ----
    #      Vᵢ^(r-1), kept in U
    for a in 1:k′
        copyto!(S.U[a], S.V[a])
    end

    # ---- the step: the k′ directions packed into one batch, block a along the points ----
    #      V = (v₁¹, …, v_m¹, v₁², …, v_m², …)
    for a in 1:k′
        copyto!(_block(S.Vk′, a, m), S.V[a])
    end

    # ---- the step: the batch advanced over the segments along the shooting points, which
    #      become the end points of the segments ----
    #      vᵢᵃ ← ψᵢ vᵢᵃ,   Xᵢ = φ_τ(uᵢ)
    copyto!(S.tmp, z.X)
    tangent!(S.P, S.tmp, S.Vk′, period(z)/nsegments(z))

    # ---- the step: each image back in its batch, at the point it reaches, the last shifted ----
    #      ψᵢ vᵢᵃ at point i + 1,   S(s) ψ_m v_mᵃ at point 1
    for a in 1:k′
        copyto!(S.V[a], _block(S.Vk′, a, m))
        nshifts(z) > 0 && shift!(S.P, _tail(S.V[a]), shifts(z))
        shift_forward!(S.V[a])
    end

    # ---- the Galerkin blocks of the bases before the step, from the images ----
    #      Eᵢ = Vᵢ₊₁^(r-1)ᵀ ψᵢ Vᵢ^(r-1)
    _galerkin!(S.E, S.U, S.V)

    # ---- the step: the QR factorisation of the images at every point ----
    #      ψᵢ Vᵢ^(r-1) = Vᵢ₊₁^(r) Rᵢ^(r)
    _orthonormalise!(S.V)

    # ---- the turn and its singular values ----
    #      Zᵢ = Vᵢ^(r) - Vᵢ^(r-1) (Vᵢ^(r-1)ᵀ Vᵢ^(r)),   σⱼ(Zᵢ) = sin ϑⱼ,ᵢ
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
- `z::MultiVector`: the new point, kept, not copied.
- `tol::Real`: the tolerance ε_sub on the sine of the turn, as [`maxturn`](@ref).
- `k::Int=k′`: the number of leading directions that the test checks, from 1 to the number `k′`
  of directions of each basis.
- `maxsteps::Int=100`: the most steps.
- `cap::Symbol=:error`: when the bases fail the test after `maxsteps` steps, `:error` throws an
  error and `:keep` keeps them as they are. With `tol = 0` and `:keep` the update makes exactly
  `maxsteps` steps.
"""
function update!(S::Subspace, z::MultiVector; tol::Real,
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
        maxturn(S; k) <= ε_sub &&
            return S
    end

    # ---- the test failed after the cap: the bases kept as they are, or an error ----
    cap === :keep &&
        return S
    error("the bases failed the test in $(maxsteps) subspace steps: the turn of the $(k) leading " *
          "directions is $(maxturn(S; k)), above $(ε_sub)")
end

# The Galerkin blocks of the bases U before a step, from their images V before Gram–Schmidt, k′
# batches each. After the shift forward the image of segment i sits at point i + 1, cyclically, and its
# coordinates in the basis there are the block of segment i,
#
#     Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E_m = U₁ᵀ S(s) ψ_m U_m,
#
# one inner product per point for every pair of directions, from dots, brought to the host
function _galerkin!(E::Array{Float64, 3}, U::Vector, V::Vector)
    k′ = length(U)

    for a in 1:k′, b in 1:k′
        # ---- at every point j, the coordinate of the image that reaches it ----
        #      ⟨uⱼᵃ, ψⱼ₋₁ uⱼ₋₁ᵇ⟩
        c = vec(Array(dots(U[a], V[b])))

        # ---- moved back to the segment the image comes from ----
        #      (Eᵢ)ₐᵦ = ⟨uᵢ₊₁ᵃ, ψᵢ uᵢᵇ⟩
        E[a, b, :] .= circshift(c, -1)
    end

    return E
end

# The turn of a step at every point, from the bases U before it and V after it, k′ batches each:
# the part of the new basis outside the old span, formed explicitly in Z,
#
#     Zᵢ = Vᵢ - Uᵢ Cᵢ,   Cᵢ = Uᵢᵀ Vᵢ,
#
# and its singular values in increasing order, written into σ. They are those of the triangular
# factor of a QR factorisation Zᵢ = Qᵢ Tᵢ by one pass of modified Gram–Schmidt, which leaves Qᵢ in
# Z. That factor is backward stable however far Qᵢ is from orthogonal, so that the small sines are
# resolved down to eps times the largest, where the eigenvalues of ZᵢᵀZᵢ would square them and lose
# them below √eps times the largest. The sines are capped at 1, against rounding
function _turn!(Z::Vector, σ::Matrix{Float64}, U::Vector, V::Vector)
    k′ = length(V)
    m  = size(σ, 2)

    # ---- the coordinates of the new basis in the old, one per point ----
    #      (Cᵢ)ₐᵦ = ⟨uᵢᵃ, vᵢᵇ⟩
    C = [dots(U[a], V[b]) for a in 1:k′, b in 1:k′]

    # ---- the new basis minus its projection on the old ----
    #      zᵢᵇ = vᵢᵇ - Σₐ (Cᵢ)ₐᵦ uᵢᵃ
    for b in 1:k′
        copyto!(Z[b], V[b])
        for a in 1:k′
            Z[b] .-= C[a, b] .* U[a]
        end
    end

    # ---- its singular values, those of its triangular factor, in increasing order ----
    #      Zᵢ = Qᵢ Tᵢ,   σⱼ(Zᵢ) = σⱼ(Tᵢ) = sin ϑⱼ,ᵢ
    T = _gram_schmidt!(Z)
    for i in 1:m
        σ[:, i] .= sort(min.(LinearAlgebra.svdvals(T[:, :, i]), 1.0))
    end

    return σ
end

# The QR factorisation of a step, the images V overwritten with the new bases: two passes of
# modified Gram–Schmidt, the second for orthogonality to rounding. The triangular factor Rᵢ^(r) is
# not needed
function _orthonormalise!(V::Vector)
    for _ in 1:2
        _gram_schmidt!(V)
    end

    return V
end

# One pass of modified Gram–Schmidt across the k′ batches, at every point at once, in the order of
# the directions, so that the first j columns of Qᵢ span the first j of Vᵢ: V is overwritten with Q
# and the triangular factor returned, Vᵢ = Qᵢ Rᵢ, k′ × k′ × m on the host. The coefficients are one
# per point, from the inner product dots of the states, and broadcast against the batches. A
# column that vanishes at a point stays zero there, with a zero on the diagonal of R
function _gram_schmidt!(V::Vector)
    k′ = length(V)
    m  = nsegments(first(V))
    R  = zeros(k′, k′, m)

    for a in 1:k′
        # ---- minus its components along the columns before it ----
        #      vᵢᵃ ← vᵢᵃ - ⟨qᵢᵇ, vᵢᵃ⟩ qᵢᵇ,   b = 1, …, a - 1
        for b in 1:a-1
            c = dots(V[b], V[a])
            V[a] .-= c .* V[b]
            R[b, a, :] .= vec(Array(c))
        end

        # ---- normalised ----
        #      qᵢᵃ = vᵢᵃ/‖vᵢᵃ‖
        n = sqrt.(dots(V[a], V[a]))
        V[a] ./= max.(n, floatmin(Float64))
        R[a, a, :] .= vec(Array(n))
    end

    return R
end
