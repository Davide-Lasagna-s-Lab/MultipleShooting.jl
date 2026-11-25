# The Jacobian of the residual, matrix free, by a finite difference of the residual.
#
# The product is the finite difference of NKRoots.jl, jacobianFD, of the residual of residual.jl,
# forward or central: one integration of the seeds per forward product, two per central one, and
# none of the tangent equation. The difference gives the columns of the period and of the shifts as
# well, as derivatives of the integrator. The residual has no scalar part, since the phase
# conditions act on the Newton step only, and its difference has none either: the phase rows of
# jacobian.jl, ⟨f(u₁), v₁⟩ in time and ⟨gⱼ(u₁), v₁⟩ in space, are added to it.

"""
    JacobianFD(P, z; difference=:forward, ϵ=nothing)

The Jacobian of the multiple-shooting residual of the propagator `P` at `z`, matrix free, by the
finite difference `jacobianFD` of NKRoots.jl of the [`Residual`](@ref), with the phase rows of
[`Jacobian`](@ref),

    J v ≈ (F(z + ε v) - F(z))/ε,   (J v)_T = ⟨f(u₁), v₁⟩,   (J v)_sⱼ = ⟨gⱼ(u₁), v₁⟩,

or the central difference, with the step ε of `jacobianFD`. A forward product integrates the seeds
once, a central one twice, and the tangent equation never. `mul!(out, J, v)` or `J*v` applies it,
`update!(J, z, r)` moves it to `z` with the residual `r` there, with no integration, and
`update!(J, z)` with one, for the residual.

# Arguments
- `P`: the propagator, with `propagate!` and `rhs!`, and `shift!` and `dshift!` for relative
  periodic orbits; no `tangent!`.
- `z::MultiVector`: the point, kept, not copied; the residual is evaluated there once.
- `difference::Symbol=:forward`: `:forward` or `:central`, as for `jacobianFD`.
- `ϵ`: the relative step of `jacobianFD`, or `nothing` for its default.
"""
mutable struct JacobianFD{PT, V<:MultiVector, D, H}
    P::PT        # the propagator
    z::V         # the point, not a copy
    D::D         # the finite difference of the residual, of NKRoots.jl, at z
    f::H         # f at the first seed, a batch of one state
    g::Vector{H} # the generators at the first seed, one per shift

    function JacobianFD(P, z::MultiVector; difference::Symbol=:forward,
                                           ϵ::Union{Real, Nothing}=nothing)
        # ---- the difference of the residual at z, which evaluates the residual once ----
        D = NKRoots.jacobianFD(Residual(P), z; difference=difference, ϵ=ϵ)

        # ---- the storage of the phase rows, filled by update! ----
        f = similar(head(z.X))
        g = typeof(f)[similar(f) for _ in 1:nshifts(z)]

        J = new{typeof(P), typeof(z), typeof(D), typeof(f)}(P,
                                                            z,
                                                            D,
                                                            f,
                                                            g)
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
- `z::MultiVector`: the new point, kept, not copied.
- `r::MultiVector`: the residual at `z`.
"""
function NKRoots.update!(J::JacobianFD, z::MultiVector, r::MultiVector)
    J.z = z

    # ---- the difference at z, with the residual there ----
    NKRoots.update!(J.D, z, r)

    # ---- f and the generators at the first seed, the phase rows ----
    rhs!(J.P, J.f, head(z.X))
    for j in 1:nshifts(z)
        dshift!(J.P, J.g[j], head(z.X), j)
    end

    return J
end

NKRoots.update!(J::JacobianFD, z::MultiVector) =
    NKRoots.update!(J, z, Residual(J.P)(similar(z), z))

function LinearAlgebra.mul!(out::MultiVector, J::JacobianFD, v::MultiVector)
    # ---- the difference of the residual: the rows of the seeds, with the period and the shifts ----
    LinearAlgebra.mul!(out, J.D, v)

    # ---- the phase conditions, in time and in space ----
    #      d₁ = ⟨f(u₁), v₁⟩,   d₁₊ⱼ = ⟨gⱼ(u₁), v₁⟩
    out.d[1] = _dot(J.f, head(v.X))
    for j in 1:nshifts(J.z)
        out.d[1 + j] = _dot(J.g[j], head(v.X))
    end

    return out
end

Base.:*(J::JacobianFD, v::MultiVector) = LinearAlgebra.mul!(similar(v), J, v)
