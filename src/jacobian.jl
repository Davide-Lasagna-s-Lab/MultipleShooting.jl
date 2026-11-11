# The Jacobian of the residual, matrix free, with the products by the tangent equation.
#
# With the reordered rows of residual.jl, the Jacobian at z applied to v = (v₁, …, v_m, δT, δs₁, …)
# has in the row of segment i + 1
#
#     ψᵢ vᵢ - vᵢ₊₁ + δτ f(φ_τ(uᵢ)),
#
# with ψᵢ the tangent map of segment i and δτ = δT/m the change of the length of a segment. In
# row 1, which closes the orbit, the end of the last segment is shifted, and the shifts add their
# columns,
#
#     S(s) ψ_m v_m - v₁ + δτ f(w) + Σⱼ δsⱼ gⱼ(w),   w = S(s) φ_τ(u_m),
#
# with gⱼ the generator of shift j, by the equivariance of the system. The scalar part holds the
# phase conditions, ⟨f(u₁), v₁⟩ in time and ⟨gⱼ(u₁), v₁⟩ in space, which keep the phase of the
# first seed. The product follows the residual step by step: the vector is advanced over the
# segments by the tangent equation, along the seeds in a batch of their own, so that a product
# integrates the seeds again; the end of the last segment is shifted, the columns of the period
# and of the shifts are added, and each row is moved into the row of the seed it has to reach
# before the vector is subtracted. f and the generators at the end points and at the first seed
# are kept, and computed again by update! at a new point, which integrates the seeds once.
#
# The column of the period comes from dpropagate!, with the end points in the same integration: by
# default f at the end points, the derivative of the exact flow, which differs from that of the
# integrator by the error of the scheme; with the method of a propagator, the derivative of its
# integrator, of which the residual is made.

"""
    Jacobian(P, z)

The Jacobian of the multiple-shooting residual of the propagator `P` at `z`, matrix free:
`mul!(out, J, v)` or `J*v` applies it to v = (v₁, …, v_m, δT, δs₁, …, δs_c),

    (Jv)ᵢ₊₁ = ψᵢ vᵢ - vᵢ₊₁ + δτ ∂φ_τ(uᵢ)/∂τ,   i = 1, …, m - 1,
    (Jv)₁   = S(s) ψ_m v_m - v₁ + δτ S(s) ∂φ_τ(u_m)/∂τ + Σⱼ δsⱼ gⱼ(w),   w = S(s) φ_τ(u_m),
    (Jv)_T  = ⟨f(u₁), v₁⟩,   (Jv)_sⱼ = ⟨gⱼ(u₁), v₁⟩,

with δτ = δT/m, ψᵢ the tangent maps, by the tangent equation along the seeds, one integration per
product, and the scalar rows the phase conditions. Building it integrates the seeds once, by
[`dpropagate!`](@ref), which also gives the column of the period.

# Arguments
- `P`: the propagator, with `propagate!`, `tangent!` and `rhs!`, `shift!` and `dshift!` for
  relative periodic orbits, and its own `dpropagate!` for the derivative of its integrator.
- `z::MultiVector`: the point, kept, not copied; `update!(J, z)` moves it to another.
"""
mutable struct Jacobian{PT, V<:MultiVector, A, H}
         P::PT        # the propagator
         z::V         # the point, not a copy
       tmp::A         # work batch: the end points, then the seeds advanced with the vectors
         F::A         # the column of the period times m: in row i, the derivative in τ of the
                      # end of segment i, by dpropagate!, the last one shifted
         G::Vector{H} # the generators at the shifted end of the last segment, one per shift
         f::H         # f at the first seed, a batch of one state
         g::Vector{H} # the generators at the first seed, one per shift

    function Jacobian(P, z::MultiVector)
        # ---- the storage, filled by update! ----
        tmp = _similar(z.X)
        F = _similar(z.X)
        f = _similar(_head(z.X))
        G = typeof(f)[_similar(f) for _ in 1:nshifts(z)]
        g = typeof(f)[_similar(f) for _ in 1:nshifts(z)]

        J = new{typeof(P), typeof(z), typeof(F), typeof(f)}(P,
                                                             z,
                                                             tmp,
                                                             F,
                                                             G,
                                                             f,
                                                             g)
        return NKRoots.update!(J, z)
    end
end

"""
    update!(J::Jacobian, z) -> J
    update!(J::Jacobian, z, r) -> J

Move `J` to the point `z`: the column of the period, and f and the generators at the end points
of the segments and at the first seed, with one integration of the seeds. The second form is the
one `nkroot!` of NKRoots.jl calls.

# Arguments
- `J::Jacobian`: the Jacobian, changed.
- `z::MultiVector`: the new point, kept, not copied.
- `r::MultiVector`: the residual at `z`, not used.
"""
function NKRoots.update!(J::Jacobian, z::MultiVector)
    τ   = period(z)/nsegments(z)
    J.z = z

    # ---- the end points, row i φ_τ(uᵢ), and their derivatives in τ, the column of the period ----
    _copyto!(J.tmp, z.X)
    dpropagate!(J.P, J.tmp, J.F, τ)

    # ---- the end of the last segment shifted, and its derivative with it ----
    if nshifts(z) > 0
        shift!(J.P, _tail(J.tmp), _shifts(z))
        shift!(J.P, _tail(J.F), _shifts(z))
    end

    # ---- f and the generators at the end of the last segment, and at the first seed ----
    rhs!(J.P, J.f, _head(z.X))
    for j in 1:nshifts(z)
        dshift!(J.P, J.G[j], _tail(J.tmp), j)
        dshift!(J.P, J.g[j], _head(z.X), j)
    end

    return J
end

# as nkroot! of NKRoots.jl calls it, at the start of every Newton iteration
NKRoots.update!(J::Jacobian, z::MultiVector, r::MultiVector) = NKRoots.update!(J, z)

function LinearAlgebra.mul!(out::MultiVector, J::Jacobian, v::MultiVector)
    τ  = period(J.z)/nsegments(v)
    δτ = period(v)/nsegments(v)

    # ---- the seeds and the vector ----
    #      tmpᵢ = uᵢ,   outᵢ = vᵢ
    _copyto!(J.tmp, J.z.X)
    out .= v

    # ---- advanced over their segments by the tangent equation ----
    #      outᵢ = ψᵢ vᵢ
    tangent!(J.P, J.tmp, out.X, τ)

    # ---- the end of the last segment shifted ----
    #      out_m = S(s) ψ_m v_m
    nshifts(J.z) > 0 && shift!(J.P, _tail(out.X), _shifts(J.z))

    # ---- plus the period column, and the shift columns at the end of the last segment ----
    #      outᵢ += δτ Fᵢ, Fᵢ = ∂φ_τ(uᵢ)/∂τ,   out_m += Σⱼ δsⱼ gⱼ(S(s) φ_τ(u_m))
    _map!((ψv, f) -> ψv + δτ*f, out.X, out.X, J.F)
    for j in 1:nshifts(J.z)
        δs = shift(v, j)
        _map!((o, g) -> o + δs*g, _tail(out.X), _tail(out.X), J.G[j])
    end

    # ---- each row in the row of the seed it has to reach, minus the vector ----
    #      outᵢ₊₁ = ψᵢ vᵢ + δτ f(φ_τ(uᵢ)) - vᵢ₊₁,   out₁ = S(s) ψ_m v_m + … - v₁
    _roll!(out.X)
    out .= out .- v

    # ---- the phase conditions, in time and in space ----
    #      d₁ = ⟨f(u₁), v₁⟩,   d₁₊ⱼ = ⟨gⱼ(u₁), v₁⟩
    out.d[1] = _dot(J.f, _head(v.X))
    for j in 1:nshifts(J.z)
        out.d[1 + j] = _dot(J.g[j], _head(v.X))
    end

    return out
end

Base.:*(J::Jacobian, v::MultiVector) = LinearAlgebra.mul!(similar(v), J, v)
