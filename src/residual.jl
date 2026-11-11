# The residual of multiple shooting.
#
# With seeds u₁, …, u_m and period T the segments have length τ = T/m, and the equations are
# φ_τ(uᵢ) - uᵢ₊₁ = 0 for i < m and S(s) φ_τ(u_m) - u₁ = 0, which closes the orbit, with S(s) the
# shift of a relative periodic orbit, the identity for a periodic one. The equation of
# segment i sits in the row of segment i + 1, that of segment m in row 1. With these reordered
# rows the Jacobian is -I plus a cyclic block subdiagonal of tangent maps. The phase condition
# acts on the Newton step only, so that the scalar part of the residual is zero.
#
# The residual follows the equations: the seeds are advanced over their segments in r, the end of
# the last segment is shifted, and each end point is moved into the row of the seed it has to
# reach, before the seeds are subtracted. r is the only storage.

"""
    Residual(P)

The multiple-shooting residual of the propagator `P`, called as `F(r, z)`, as `nkroot!` of
NKRoots.jl calls it. In the row of segment i + 1 it holds the end of segment i minus the seed of
segment i + 1, and in row 1 the end of the last segment, shifted by the shifts of `z`, minus the
first seed,

    rᵢ₊₁ = φ_τ(uᵢ) - uᵢ₊₁,   i = 1, …, m - 1,   r₁ = S(s) φ_τ(u_m) - u₁,   τ = T/m,

and its scalar part is zero. One integration of the seeds per call.

# Arguments
- `P`: the propagator, with `propagate!`, and `shift!` for relative periodic orbits.

Called as `F(r, z)`, with
- `r::MultiVector`: overwritten with the residual;
- `z::MultiVector`: the unknowns, not changed.
"""
struct Residual{PT}
    P::PT # the propagator
end

function (F::Residual)(r::MultiVector, z::MultiVector)
    τ = period(z)/nsegments(z)

    # ---- the seeds ----
    #      rᵢ = uᵢ
    r .= z

    # ---- advanced over their segments ----
    #      rᵢ = φ_τ(uᵢ)
    propagate!(F.P, r.X, τ)

    # ---- the end of the last segment shifted ----
    #      r_m = S(s) φ_τ(u_m)
    nshifts(z) > 0 && shift!(F.P, _tail(r.X), _shifts(z))

    # ---- each end point in the row of the seed it has to reach, minus the seeds ----
    #      rᵢ₊₁ = φ_τ(uᵢ) - uᵢ₊₁,   r₁ = S(s) φ_τ(u_m) - u₁,   and a zero scalar part
    _roll!(r.X)
    r .= r .- z

    return r
end
