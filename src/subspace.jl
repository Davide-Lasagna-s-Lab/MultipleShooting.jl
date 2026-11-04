# Subspace iteration around the loop. At every seed i an orthonormal basis Vᵢ of k directions; a
# subspace step carries each basis over its segment by the tangent map, shifts the bases of the
# last segment, and orthonormalises them at the next seed,
#
#     Vᵢ₊₁ ← orth(ψᵢ Vᵢ),   V₁ ← orth(S(s) ψ_m V_m),
#
# all the seeds at once. On a fixed loop the part of a leading direction j outside the span falls
# by about e^{-(λⱼ - λ_{k+1}) τ} per step, with λ the Floquet exponents.
#
# The bases are k batches like the seeds, one per direction: V[a] holds direction a at every seed,
#
#     V[1] = (v₁¹, v₂¹, …, v_m¹),   …,   V[k] = (v₁ᵏ, v₂ᵏ, …, v_mᵏ),
#
# so that each is advanced, shifted and rolled as a vector of the Jacobian is, by tangent! along
# the seeds, one direction at a time. Gram–Schmidt then runs across the k batches, state by state, in order, so that the
# first p directions at every seed are a subspace iteration of their own.
#
# The accuracy of the bases is measured as in Lust (1997, sections 2.4.4 and 6.3.3): the part of
# the image of a basis that falls outside the basis at the seed it reaches,
#
#     Zᵢ₊₁ = (I - Uᵢ₊₁ Uᵢ₊₁ᵀ) Wᵢ₊₁,   Wᵢ₊₁ = ψᵢ Uᵢ,   W₁ = S(s) ψ_m U_m,
#
# with U the bases before a step and W their images, zero when the spans are invariant. Its
# largest singular value comes from k × k matrices only, ZᵀZ = WᵀW - (UᵀW)ᵀ(UᵀW). Lust measures it
# at the seed which closes the loop, where his sweep meets its start; here all the seeds advance
# together, and the largest value over the seeds is taken.

"""
    Subspace(J; k)

Orthonormal bases of `k` directions at every seed, random at the start, for subspace iteration
along the loop of the point of the Jacobian `J`, with its propagator: the bases follow `J` when it
moves. The bases are `k` batches like the seeds, one per direction. `iterate!(S)` makes one
subspace step, and `change(S)` gives the accuracy of the bases before it.
"""
struct Subspace{JT<:Jacobian, A}
      J::JT                 # the Jacobian: its propagator and its point give the tangent maps
      V::Vector{A}          # the bases, one batch per direction: V[a] holds direction a at every seed
      U::Vector{A}          # the bases before the last step
    ZᵀZ::Array{Float64, 3}  # ZᵢᵀZᵢ at every seed, k × k × m, from the last step: Inf before any
    tmp::A                  # work batch: the seeds, advanced with each direction

    function Subspace(J::Jacobian; k::Int)
        # ---- input checks ----
        k >= 1 ||
            throw(ArgumentError("k must be at least 1"))

        # ---- random orthonormal bases ----
        X = J.z.X
        V = [_randn!(_similar(X)) for _ in 1:k]
        _orthonormalise!(V)

        return new{typeof(J), typeof(X)}(J,
                                         V,
                                         _similar.(V),
                                         fill(Inf, k, k, nsegments(X)),
                                         _similar(X))
    end
end

"""
    iterate!(S::Subspace) -> S

One subspace step at the current point of the Jacobian: each basis carried over its segment by
the tangent map, the bases of the last seed shifted, and all of them moved to the next seed and
orthonormalised there. On the way, the images of the bases give the matrices ZᵢᵀZᵢ of Lust's
measure of accuracy, read by [`change`](@ref). A step integrates the seeds k times, once with
each direction.
"""
function iterate!(S::Subspace)
    z = S.J.z
    k = length(S.V)
    τ = period(z)/nsegments(z)

    # ---- the bases before the step ----
    #      Uᵃ = Vᵃ
    for a in 1:k
        _copyto!(S.U[a], S.V[a])
    end

    for a in 1:k
        # ---- advanced over the segments, along a copy of the seeds ----
        #      vᵢᵃ = ψᵢ vᵢᵃ
        _copyto!(S.tmp, z.X)
        tangent!(S.J.P, S.tmp, S.V[a], τ)

        # ---- the end of the last segment shifted ----
        #      v_mᵃ = S(s) ψ_m v_mᵃ
        nshifts(z) > 0 && shift!(S.J.P, _tail(S.V[a]), _shifts(z))

        # ---- each vector at the seed it reaches ----
        #      vᵢ₊₁ᵃ = ψᵢ vᵢᵃ,   v₁ᵃ = S(s) ψ_m v_mᵃ
        _roll!(S.V[a])
    end

    # ---- Lust's residual, from the images and the bases before the step, at every seed ----
    #      ZᵢᵀZᵢ = WᵢᵀWᵢ - (UᵢᵀWᵢ)ᵀ(UᵢᵀWᵢ),   Wᵢ = Vᵢ here
    _residual!(S.ZᵀZ, S.U, S.V)

    # ---- the bases orthonormalised at every seed ----
    #      Vᵢ = orth(Vᵢ)
    _orthonormalise!(S.V)

    return S
end

# ZᵢᵀZᵢ = WᵢᵀWᵢ - (UᵢᵀWᵢ)ᵀ(UᵢᵀWᵢ) at every seed i, into the k × k × m array ZᵀZ, from the bases U
# before the step and their images W, k batches each like the seeds, through the inner products
# between the batches, brought to the host
function _residual!(ZᵀZ::Array{Float64, 3},
                      U::Vector{A},
                      W::Vector{A}) where {A}
    k = length(U)
    m = nsegments(W[1])

    # ---- the inner products, one per seed ----
    #      WᵀW[a, b][i] = ⟨wᵢᵃ, wᵢᵇ⟩,   UᵀW[a, b][i] = ⟨uᵢᵃ, wᵢᵇ⟩
    WᵀW = [Array(_dots(W[a], W[b])) for a in 1:k, b in 1:k]
    UᵀW = [Array(_dots(U[a], W[b])) for a in 1:k, b in 1:k]

    # ---- assembled seed by seed ----
    for i in 1:m
        Cᵢ = getindex.(UᵀW, i)
        ZᵀZ[:, :, i] .= getindex.(WᵀW, i) .- Cᵢ'*Cᵢ
    end

    return ZᵀZ
end

# Modified Gram–Schmidt across the batches, at every seed i, in the order of the directions, and
# twice for orthogonality to rounding. The coefficients are one per seed, from _dots, and broadcast
# against the batches in both layouts: batched, along the last dimension; sliced, state by state
function _orthonormalise!(V::Vector)
    k = length(V)

    # ---- twice, the second pass for orthogonality to rounding ----
    for _ in 1:2
        # ---- the k directions in order, at every seed at once ----
        for a in 1:k
            # ---- minus its components along the directions before it ----
            #      vᵢᵃ = vᵢᵃ - ⟨vᵢᵇ, vᵢᵃ⟩ vᵢᵇ,   b = 1, …, a - 1
            for b in 1:a-1
                c = _dots(V[b], V[a])
                V[a] .-= c .* V[b]
            end

            # ---- normalised ----
            #      vᵢᵃ = vᵢᵃ/‖vᵢᵃ‖
            n = sqrt.(_dots(V[a], V[a]))
            V[a] ./= n
        end
    end

    return V
end

"""
    change(S::Subspace; p=k)

The accuracy of the bases before the last subspace step, as in Lust (1997, sections 2.4.4 and
6.3.3): the largest singular value of Zᵢ = (I - Uᵢ Uᵢᵀ) Wᵢ restricted to the first `p`
directions, the largest over the seeds,

    ε_p = maxᵢ σ_max(Zᵢ[:, 1:p]) = maxᵢ √λ_max(ZᵢᵀZᵢ[1:p, 1:p]),

with Uᵢ the basis at seed i and Wᵢ the image of the basis of the seed before. With `p` the number
of unstable and neutral directions it is the error of these directions; `p` should not split a
complex pair of multipliers. It is absolute, not relative to the size of the images, and from
ZᵀZ it resolves down to about √eps ‖Wᵢ‖. Inf before the first step.
"""
function change(S::Subspace; p::Int=length(S.V))
    k = length(S.V)
    m = nsegments(S.J.z)

    # ---- input checks ----
    1 <= p <= k ||
        throw(ArgumentError("p must be between 1 and k"))

    # ---- nothing to measure before the first step ----
    all(isfinite, S.ZᵀZ) ||
        return Inf

    # ---- the largest singular value at every seed, the largest over the seeds ----
    #      ε_p = maxᵢ √λ_max(ZᵢᵀZᵢ[1:p, 1:p]), the rounding of ZᵀZ below zero cut off
    ε = [sqrt(max(eigmax(Symmetric(S.ZᵀZ[1:p, 1:p, i])), 0.0)) for i in 1:m]

    return maximum(ε)
end
