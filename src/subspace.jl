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
# the seeds, one direction at a time. Gram–Schmidt then runs across the k batches, state by state,
# in order, so that the first p directions at every seed are a subspace iteration of their own.
#
# The accuracy of the first p directions is measured as in Lust (1997, sections 2.4.4 and 6.3.3):
# the part of their images that falls outside their span at the seed they reach,
#
#     Zᵢ₊₁ = (I - Uᵢ₊₁ Uᵢ₊₁ᵀ) Vᵢ₊₁,   Vᵢ₊₁ = ψᵢ Uᵢ,   V₁ = S(s) ψ_m U_m,
#
# with U the first p directions of the bases before a step and V their images, before Gram–Schmidt,
# zero when the spans are invariant. The step forms Z = (I - UUᵀ)V for all k directions, and keeps
# the inner products ZᵀZ and the coordinates UᵀV of the images in the bases before; those of the
# first p follow from their blocks, as a sum with no cancellation, so that the measure resolves down
# to rounding. Lust measures it at the seed which closes the loop, where his sweep meets its start;
# here all the seeds advance together, and the largest value over the seeds is taken.

"""
    Subspace(P, z; k)

Orthonormal bases of `k` directions at every seed, random at the start, for subspace iteration
along the loop of the propagator `P` at `z`. The bases are `k` batches like the seeds, one per
direction. [`iterate!`](@ref) makes one subspace step, `update!(S, z; tol)` moves the subspace to
`z` and makes steps there until the bases are accurate, and [`change`](@ref) gives the accuracy
of the bases before the last step.

# Arguments
- `P`: the propagator, with `tangent!`, and `shift!` for relative periodic orbits.
- `z::MultiVector`: the point, kept, not copied.
- `k::Int`: the number of directions per seed, at least 1.
"""
mutable struct Subspace{PT, ZT<:MultiVector, A}
      P::PT                 # the propagator
      z::ZT                 # the point, not a copy: its seeds, period and shifts give the tangent maps
      V::Vector{A}          # the bases, one batch per direction: V[a] holds direction a at every seed
      U::Vector{A}          # the bases before the last step
      Z::Vector{A}          # the parts of the images outside the bases before, Z = (I - UUᵀ)V
    ZᵀZ::Array{Float64, 3}  # ⟨zᵢᵃ, zᵢᵇ⟩ at every seed, k × k × m, from the last step: Inf before any
    UᵀV::Array{Float64, 3}  # ⟨uᵢᵃ, vᵢᵇ⟩ at every seed, k × k × m, from the last step: Inf before any
    tmp::A                  # work batch: the seeds advanced, the end points after a step
  steps::Int                # subspace steps of the last update!

    function Subspace(P, z::MultiVector; k::Int)
        # ---- input checks ----
        k >= 1 ||
            throw(ArgumentError("k must be at least 1"))

        # ---- random orthonormal bases ----
        X = z.X
        V = [_randn!(_similar(X)) for _ in 1:k]
        _orthonormalise!(V, z.dots)

        return new{typeof(P), typeof(z), typeof(X)}(P,
                                                    z,
                                                    V,
                                                    _similar.(V),
                                                    _similar.(V),
                                                    fill(Inf, k, k, nsegments(X)),
                                                    fill(Inf, k, k, nsegments(X)),
                                                    _similar(X),
                                                    0)
    end
end

"""
    update!(S::Subspace, z; tol, maxsteps=100, p=k, cap=:error, by=:change) -> S

Move `S` to the point `z`, with no integration, and make subspace steps there from the bases it
has, at least one, until the bases before the last step are accurate,

    change(S; p) ≤ tol,   or   turn(S; p) ≤ tol with `by = :turn`,

or `maxsteps` steps have been made. A function of this package, not the `update!` of NKRoots.jl.

# Arguments
- `S::Subspace`: the subspace, changed; `S.steps` holds the number of steps made.
- `z::MultiVector`: the new point, kept, not copied.
- `tol::Real`: the accuracy wanted, absolute, as [`change`](@ref).
- `maxsteps::Int=100`: the most steps.
- `p::Int=k`: the number of leading directions whose accuracy is tested.
- `cap::Symbol=:error`: when the bases are not accurate after `maxsteps` steps, `:error` throws
  an error and `:keep` keeps them as they are. With `tol = 0` and `:keep` the update makes
  exactly `maxsteps` steps.
- `by::Symbol=:change`: the measure of the accuracy, [`change`](@ref), the part of the images of
  the first `p` directions outside their span, or [`turn`](@ref), the turn of the whole span,
  which needs no gap after the first `p` directions.
"""
function update!(S::Subspace, z::MultiVector; tol::Real,
                                              maxsteps::Int=100,
                                              p::Int=length(S.V),
                                              cap::Symbol=:error,
                                              by::Symbol=:change)
    # ---- input checks ----
    maxsteps >= 1 ||
        throw(ArgumentError("maxsteps must be at least 1"))
    cap in (:error, :keep) ||
        throw(ArgumentError("cap must be :error or :keep"))
    by in (:change, :turn) ||
        throw(ArgumentError("by must be :change or :turn"))
    accuracy() = by === :change ? change(S; p=p) : turn(S; p=p)

    # ---- the subspace moved to z, with no integration ----
    S.z = z

    # ---- steps until the bases before the last one are accurate ----
    S.steps = 0
    while S.steps < maxsteps
        iterate!(S)
        S.steps += 1
        accuracy() <= tol && return S
    end

    # ---- not accurate after the cap: kept as they are, or an error ----
    cap === :keep && return S
    error("the subspace did not reach the tolerance $(tol) in $(maxsteps) steps: $(by) " *
          "$(accuracy())")
end

"""
    iterate!(S::Subspace) -> S

One subspace step at the point of `S`: each basis carried over its segment by the tangent map,
the bases of the last seed shifted, and all of them moved to the next seed and orthonormalised
there,

    Vᵢ₊₁ ← orth(ψᵢ Vᵢ),   i = 1, …, m - 1,   V₁ ← orth(S(s) ψ_m V_m),

with Gram–Schmidt in the order of the directions, so that the spans of the first p are nested.
On the way, the images of the bases and the bases before the step give Lust's remainders
Z = (I - UUᵀ)V, read by [`change`](@ref). A step integrates the seeds k times, once with each
direction, and leaves the end points of the segments, φ_τ(uᵢ), in `S.tmp`.

# Arguments
- `S::Subspace`: the subspace, changed.
"""
function iterate!(S::Subspace)
    z = S.z
    k = length(S.V)
    τ = times(z)

    # ---- the bases before the step ----
    #      Uᵃ = Vᵃ
    for a in 1:k
        _copyto!(S.U[a], S.V[a])
    end

    for a in 1:k
        # ---- advanced over the segments, along a copy of the seeds ----
        #      vᵢᵃ = ψᵢ vᵢᵃ
        _copyto!(S.tmp, z.X)
        tangent!(S.P, S.tmp, S.V[a], τ)

        # ---- the end of the last segment shifted ----
        #      v_mᵃ = S(s) ψ_m v_mᵃ
        nshifts(z) > 0 && shift!(S.P, _tail(S.V[a]), _shifts(z))

        # ---- each vector at the seed it reaches ----
        #      vᵢ₊₁ᵃ = ψᵢ vᵢᵃ,   v₁ᵃ = S(s) ψ_m v_mᵃ
        _roll!(S.V[a])
    end

    # ---- Lust's remainders, from the images and the bases before the step, at every seed ----
    #      Zᵢ = (I - UᵢUᵢᵀ)Vᵢ,   with Vᵢ the images, not yet orthonormalised
    _remainder!(S.Z, S.ZᵀZ, S.UᵀV, S.U, S.V, z.dots)

    # ---- the bases orthonormalised at every seed ----
    #      Vᵢ = orth(Vᵢ)
    _orthonormalise!(S.V, z.dots)

    return S
end

# At every seed i, the coordinates UᵢᵀVᵢ of the images in the bases before the step, the remainders
# Zᵢ = Vᵢ - Uᵢ(UᵢᵀVᵢ) = (I - UᵢUᵢᵀ)Vᵢ, and their inner products ZᵢᵀZᵢ, from the bases U before the
# step and their images V, k batches each like the seeds. The coordinates are one per seed, from
# the inner product dots of the states, and broadcast against the batches; the inner products are
# brought to the host
function _remainder!(   Z::Vector{A},
                      ZᵀZ::Array{Float64, 3},
                      UᵀV::Array{Float64, 3},
                        U::Vector{A},
                        V::Vector{A},
                     dots) where {A}
    k = length(U)

    # ---- the coordinates of the images in the bases before ----
    #      UᵀV[a, b, i] = ⟨uᵢᵃ, vᵢᵇ⟩
    c = [dots(U[a], V[b]) for a in 1:k, b in 1:k]
    for a in 1:k, b in 1:k
        UᵀV[a, b, :] .= vec(Array(c[a, b]))
    end

    # ---- the remainders ----
    #      zᵢᵇ = vᵢᵇ - Σₐ ⟨uᵢᵃ, vᵢᵇ⟩ uᵢᵃ
    for b in 1:k
        _copyto!(Z[b], V[b])
        for a in 1:k
            Z[b] .-= c[a, b] .* U[a]
        end
    end

    # ---- their inner products ----
    #      ZᵀZ[a, b, i] = ⟨zᵢᵃ, zᵢᵇ⟩
    for a in 1:k, b in 1:k
        ZᵀZ[a, b, :] .= vec(Array(dots(Z[a], Z[b])))
    end

    return Z
end

# Modified Gram–Schmidt across the batches, at every seed i, in the order of the directions, and
# twice for orthogonality to rounding. The coefficients are one per seed, from the inner product
# dots of the states, and broadcast against the batches in both layouts: batched, along the last
# dimension; sliced, state by state
function _orthonormalise!(V::Vector, dots)
    k = length(V)

    # ---- twice, the second pass for orthogonality to rounding ----
    for _ in 1:2
        # ---- the k directions in order, at every seed at once ----
        for a in 1:k
            # ---- minus its components along the directions before it ----
            #      vᵢᵃ = vᵢᵃ - ⟨vᵢᵇ, vᵢᵃ⟩ vᵢᵇ,   b = 1, …, a - 1
            for b in 1:a-1
                c = dots(V[b], V[a])
                V[a] .-= c .* V[b]
            end

            # ---- normalised ----
            #      vᵢᵃ = vᵢᵃ/‖vᵢᵃ‖
            n = sqrt.(dots(V[a], V[a]))
            V[a] ./= n
        end
    end

    return V
end

"""
    change(S::Subspace; p=k)

The accuracy of the first `p` directions of the bases before the last subspace step, as in Lust
(1997, sections 2.4.4 and 6.3.3): the largest singular value of the part of their images that
falls outside their span, the largest over the seeds,

    ε_p = maxᵢ σ_max(Zᵢ) = maxᵢ √λ_max(ZᵢᵀZᵢ),   Zᵢ = (I - Uᵢ Uᵢᵀ) Vᵢ,

with Uᵢ the first `p` directions of the basis at seed i and Vᵢ the images of those of the seed
before. The largest singular value of Zᵢ, an N × p matrix spread over the batches, is computed as
the square root of the largest eigenvalue of ZᵢᵀZᵢ, a p × p matrix of inner products.

# Arguments
- `S::Subspace`: the subspace, after at least one step.
- `p::Int=k`: the number of leading directions measured, from 1 to k; it should not split a
  complex pair of multipliers.

[`iterate!`](@ref) forms the remainder Z = (I - UUᵀ)V of all k directions and keeps ZᵀZ and the
coordinates UᵀV of the images in the bases before, at every seed. The remainder of the first `p`
directions is not formed: with the directions split into the first `p` and the others,
r = p + 1, …, k,

    Zₚ = (I - UₚUₚᵀ)Vₚ = Z[:, 1:p] + U[:, r] (UᵀV)[r, 1:p],

the remainder of all k plus the part along the other directions. The two parts are orthogonal,
since Z is orthogonal to every column of U, so that

    ZₚᵀZₚ = (ZᵀZ)[1:p, 1:p] + (UᵀV)[r, 1:p]ᵀ (UᵀV)[r, 1:p],

a sum of two positive semidefinite blocks of the matrices kept, with no cancellation.

With `p` the number of unstable and neutral directions it is the error of these directions. It
is absolute, not relative to the size of the images, and since Z is formed it resolves down to
about eps ‖Vᵢ‖. Inf before the first step.
"""
function change(S::Subspace; p::Int=length(S.V))
    k = length(S.V)
    m = nsegments(S.z)

    # ---- input checks ----
    1 <= p <= k ||
        throw(ArgumentError("p must be between 1 and k"))

    # ---- nothing to measure before the first step ----
    all(isfinite, S.ZᵀZ) ||
        return Inf

    # ---- the largest singular value at every seed, from the blocks, as in the docstring ----
    #      ZₚᵀZₚ = (ZᵀZ)[1:p, 1:p] + (UᵀV)[r, 1:p]ᵀ (UᵀV)[r, 1:p],   r = p + 1, …, k
    r = (p + 1):k
    ε = zeros(m)
    for i in 1:m
        UᵀV  = S.UᵀV[r, 1:p, i]
        ZᵀZ  = S.ZᵀZ[1:p, 1:p, i] .+ UᵀV'*UᵀV
        ε[i] = sqrt(max(LinearAlgebra.eigmax(LinearAlgebra.Symmetric(ZᵀZ)), 0.0))
    end

    # ---- the largest over the seeds ----
    return maximum(ε)
end

"""
    turn(S::Subspace; p=k)

How far the span of the bases turned in the last subspace step: the sine of the `p`-th smallest
principal angle between the span of the k directions before the step and the span of their
images, at every seed, the largest over the seeds. When the bases hold `p` directions that the
tangent maps carry into the span, as the leading directions once they have settled, the `p`
smallest angles vanish, whatever the other directions do, so that the measure needs no gap
between the first `p` directions and the next, as [`change`](@ref) does.

# Arguments
- `S::Subspace`: the subspace, after at least one step.
- `p::Int=k`: the angle measured, from 1, the smallest, to k, the largest.

At seed i, with Uᵢ the bases before the step and Vᵢ the images that reach it, the principal angles
come from the matrices that the step keeps, Aᵢ = UᵢᵀVᵢ and ZᵢᵀZᵢ, with no inner product of the
batches: since Vᵢ = UᵢAᵢ + Zᵢ with Zᵢ orthogonal to Uᵢ, the Gram matrix of the images is
Gᵢ = AᵢᵀAᵢ + ZᵢᵀZᵢ, and the squared sines of the angles are the eigenvalues of the pencil
(ZᵢᵀZᵢ, Gᵢ), those of the part of the orthonormalised images outside the span of Uᵢ. Inf before the
first step.
"""
function turn(S::Subspace; p::Int=length(S.V))
    k = length(S.V)
    m = nsegments(S.z)

    # ---- input checks ----
    1 <= p <= k ||
        throw(ArgumentError("p must be between 1 and k"))

    # ---- nothing to measure before the first step ----
    all(isfinite, S.ZᵀZ) ||
        return Inf

    # ---- the squared sines at every seed, from the pencil (ZᵀZ, AᵀA + ZᵀZ), smallest first ----
    s = zeros(m)
    for i in 1:m
        A    = S.UᵀV[:, :, i]
        ZᵀZ  = LinearAlgebra.Symmetric(S.ZᵀZ[:, :, i])
        G    = LinearAlgebra.Symmetric(A'*A .+ ZᵀZ)
        λ    = LinearAlgebra.eigvals(ZᵀZ, G)
        s[i] = sqrt(clamp(sort(λ)[p], 0.0, 1.0))
    end

    # ---- the largest over the seeds ----
    return maximum(s)
end
