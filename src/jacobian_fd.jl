# The Jacobian of the residual, matrix free, by a finite difference of the residual.
#
# The product is the finite difference of NKRoots.jl, jacobianFD, of the residual of residual.jl,
# forward or central: one integration of the seeds per forward product, two per central one, and
# none of the tangent equation. The difference gives the columns of the period and of the shifts as
# well, as derivatives of the integrator. The residual has no scalar part, since the phase
# conditions act on the Newton step only, and its difference has none either: the phase rows of
# jacobian.jl, ⟨f(u₁), v₁⟩ in time and ⟨gⱼ(u₁), v₁⟩ in space, are added to it.

"""
    JacobianFD(G, F, z; S=nothing, dS=nothing, difference=:forward, ϵ=nothing)

The Jacobian of the multiple-shooting residual of the flow `G` at `z`, matrix free, by the
finite difference `jacobianFD` of NKRoots.jl of the [`Residual`](@ref), with the phase rows of
[`Jacobian`](@ref),

    J v ≈ (F(z + ε v) - F(z))/ε,   (J v)_T = ⟨f(u₁), v₁⟩,   (J v)_sⱼ = ⟨gⱼ(u₁), v₁⟩,

or the central difference, with the step ε of `jacobianFD`. A forward product integrates the seeds
once, a central one twice, and the tangent equation never. `mul!(out, J, v)` or `J*v` applies it,
`update!(J, z, r)` moves it to `z` with the residual `r` there, with no integration, and
`update!(J, z)` with one, for the residual.

# Arguments
- `G`: the flow, called as `G(X, τ)`; no tangent.
- `F`: the right-hand side, called as `F(dX, X)`, for the phase condition in time.
- `z::ShootingVector`: the point, kept, not copied; the residual is evaluated there once.
- `S=nothing`, `dS=nothing`: the shift and its generator, as for the [`Jacobian`](@ref), for
  relative periodic orbits.
- `difference::Symbol=:forward`: `:forward` or `:central`, as for `jacobianFD`.
- `ϵ`: the relative step of `jacobianFD`, or `nothing` for its default.
"""
mutable struct JacobianFD{V<:ShootingVector, D, A, H}
     R::Residual        # the residual, of the flow and the shift
     F::Any             # the right-hand side
    dS::Any             # the generator of the shift, or nothing
     z::V               # the point, not a copy
     D::D               # the finite difference of the residual, of NKRoots.jl, at z
     f::A               # f at the seeds, for the phase condition at the first: computed at all of
                        # them, so that F sees batches of m states only
    g₁::Vector{H}       # the generators at the first seed, one per shift

    function JacobianFD(         G,
                                 F,
                                 z::ShootingVector;
                                 S=nothing,
                                dS=nothing,
                        difference::Symbol=:forward,
                                 ϵ::Union{Real, Nothing}=nothing)
        # ---- input checks ----
        nshifts(z) > 0 && (S === nothing || dS === nothing) &&
            throw(ArgumentError("a relative periodic orbit needs its shift S and its generator dS"))

        # ---- the difference of the residual at z, which evaluates the residual once ----
        R = Residual(G; S=S)
        D = NKRoots.jacobianFD(R, z; difference=difference, ϵ=ϵ)

        # ---- the storage of the phase rows, filled by update! ----
        f = similar(z.X)
        h = similar(head(z.X))
        g₁ = typeof(h)[similar(h) for _ in 1:nshifts(z)]

        J = new{typeof(z), typeof(D), typeof(f), typeof(h)}(R,
                                                             F,
                                                             dS,
                                                             z,
                                                             D,
                                                             f,
                                                             g₁)
        return NKRoots.update!(J, z, D.Fx)
    end
end

"""
    update!(J::JacobianFD, z, r) -> J
    update!(J::JacobianFD, z) -> J

Move `J` to the point `z`: the difference to `z`, with the residual `r` there, and f and the
generators at the first seed, with no integration. The first form is the one `nkroot!` of
NKRoots.jl calls; the second evaluates the residual, with one integration of the seeds.

# Arguments
- `J::JacobianFD`: the Jacobian, changed.
- `z::ShootingVector`: the new point, kept, not copied.
- `r::ShootingVector`: the residual at `z`.
"""
function NKRoots.update!(J::JacobianFD, z::ShootingVector, r::ShootingVector)
    J.z = z

    # ---- the difference at z, with the residual there ----
    NKRoots.update!(J.D, z, r)

    # ---- f at all the seeds, of which the phase rows take the first, and the generators at
    #      the first seed ----
    J.F(J.f, z.X)
    for j in 1:nshifts(z)
        J.dS(J.g₁[j], head(z.X), j)
    end

    return J
end

NKRoots.update!(J::JacobianFD, z::ShootingVector) =
    NKRoots.update!(J, z, J.R(similar(z), z))

function LinearAlgebra.mul!(out::ShootingVector, J::JacobianFD, v::ShootingVector)
    # ---- the difference of the residual: the rows of the seeds, with the period and the shifts ----
    LinearAlgebra.mul!(out, J.D, v)

    # ---- the phase conditions, in time and in space ----
    #      d₁ = ⟨f(u₁), v₁⟩,   d₁₊ⱼ = ⟨gⱼ(u₁), v₁⟩
    out.d[1] = _dot(head(J.f), head(v.X))
    for j in 1:nshifts(J.z)
        out.d[1 + j] = _dot(J.g₁[j], head(v.X))
    end

    return out
end

Base.:*(J::JacobianFD, v::ShootingVector) = LinearAlgebra.mul!(similar(v), J, v)
