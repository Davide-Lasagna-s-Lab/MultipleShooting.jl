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
    forced!(P, X, V, τ) -> V

As [`tangent!`](@ref), with the tangent equation forced by the derivative of the right-hand side
in the parameter, `v' = ∂f/∂x(x) v + ∂f/∂p(x)`. From `V = 0` it gives the derivative of the end
points in the parameter. For [`tangent_sensitivity`](@ref) only.
"""
function forced! end

"""
    tangent_sensitivity(P, z; tol=1e-10, maxiter=200) -> dz

The derivative `dz/dp` of the converged orbit `z` of the propagator `P` in the parameter of the
equation, as a [`MultiVector`](@ref): the derivatives of the seeds, of the period and of the
shifts. Those of the period and of the shifts do not depend on the phase of the orbit; those of
the seeds do, and keep the phase conditions at `z`. The Jacobian is built at `z`, with one
integration of the seeds, the right-hand side is one more by the propagator's `forced!`, and the
system is solved by GMRES to the relative residual `tol`.
"""
function tangent_sensitivity(P, z::MultiVector; tol::Real=1e-10, maxiter::Int=200)
    J = Jacobian(P, z)
    τ = period(z)/nsegments(z)

    # ---- the seeds, and vectors from zero ----
    #      tmpᵢ = uᵢ,   bᵢ = 0
    b = similar(z)
    _copyto!(J.tmp, z.X)
    b .= 0

    # ---- advanced over their segments by the forced tangent equation ----
    #      bᵢ = ∂φ_τ(uᵢ)/∂p
    forced!(P, J.tmp, b.X, τ)

    # ---- the end of the last segment shifted ----
    #      b_m = S(s) ∂φ_τ(u_m)/∂p
    nshifts(z) > 0 && shift!(P, _tail(b.X), _shifts(z))

    # ---- each row in the row of the seed it has to reach, with the sign of -∂F/∂p ----
    #      bᵢ₊₁ = -∂φ_τ(uᵢ)/∂p,   b₁ = -S(s) ∂φ_τ(u_m)/∂p,   and a zero scalar part
    _roll!(b.X)
    b .= .-b

    # ---- J dz/dp = -∂F/∂p ----
    return _gmres((out, v) -> mul!(out, J, v), b, tol, maxiter)
end
