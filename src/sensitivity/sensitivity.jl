# Sensitivity of a periodic or relative periodic orbit to a parameter p of the equation. The
# residual vanishes along the family of orbits, F(z(p), p) = 0, so that
#
#     J dz/dp = -∂F/∂p,
#
# with J the Jacobian at the orbit. With the reordered rows of residual.jl, the row of segment
# i + 1 of ∂F/∂p is the derivative in p of the end of segment i, the solution at τ of the tangent
# equation forced by the derivative of the right-hand side in p,
#
#     v' = ∂f/∂x v + ∂f/∂p,   v(0) = 0,
#
# the last one shifted, and the phase rows have none. The scalar part of dz/dp holds dT/dp and the
# derivatives of the shifts. The right-hand side costs one forced integration of the seeds, and
# the system is solved by GMRES (gmres.jl), one product with the Jacobian per iteration.

"""
    tangent_sensitivity(P, z; tol=1e-10, maxiter=200) -> dz

The derivative `dz/dp` of the converged orbit `z` of the propagator `P` in the parameter of the
equation, as a [`MultiVector`](@ref): the derivatives of the seeds, of the period and of the
shifts, the solution of

    J dz/dp = -∂F/∂p,   (∂F/∂p)ᵢ₊₁ = ∂φ_τ(uᵢ)/∂p,   (∂F/∂p)₁ = S(s) ∂φ_τ(u_m)/∂p,

with J the [`Jacobian`](@ref) at the orbit and the derivatives of the end points from
[`forced!`](@ref). Those of the period and of the shifts do not depend on the phase of the orbit;
those of the seeds do, and keep the phase conditions at `z`. The Jacobian is built at `z`, with one
integration of the seeds, the right-hand side is one more by `forced!`, and the system is solved
by GMRES.

# Arguments
- `P`: the propagator, with `forced!` beside the functions of the [`Jacobian`](@ref).
- `z::MultiVector`: a converged orbit, not changed.
- `tol::Real=1e-10`: the relative residual of GMRES.
- `maxiter::Int=200`: the most iterations of GMRES.
"""
function tangent_sensitivity(P, z::MultiVector; tol::Real=1e-10, maxiter::Int=200)
    J = Jacobian(P, z)

    # ---- the right-hand side, -∂F/∂p ----
    b  = _forced(P, z)
    b .= .-b

    # ---- J dz/dp = -∂F/∂p ----
    return gmres((out, v) -> LinearAlgebra.mul!(out, J, v), b; tol=tol, maxiter=maxiter).x
end

# the derivative ∂F/∂p of the residual of the propagator P at z in the parameter of the equation,
# at fixed seeds, period and shifts: one integration of the seeds by the forced tangent equation.
# The shifts are taken not to depend on the parameter
function _forced(P, z::MultiVector)
    X = copy(z.X)

    # ---- vectors from zero ----
    #      cᵢ = 0
    c  = similar(z)
    c .= 0

    # ---- advanced over their segments by the forced tangent equation ----
    #      cᵢ = ∂φ_τ(uᵢ)/∂p
    forced!(P, X, c.X, period(z)/nsegments(z))

    # ---- the end of the last segment shifted ----
    #      c_m = S(s) ∂φ_τ(u_m)/∂p
    nshifts(z) > 0 && shift!(P, _tail(c.X), shifts(z))

    # ---- each row in the row of the seed it has to reach ----
    #      cᵢ₊₁ = ∂φ_τ(uᵢ)/∂p,   c₁ = S(s) ∂φ_τ(u_m)/∂p,   and a zero scalar part
    shift_forward!(c.X)

    return c
end
