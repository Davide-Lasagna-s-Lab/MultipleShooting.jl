# The residual of multiple shooting.
#
# With seeds u₁, …, u_m and period T the segments have the same length τ = T/m, and the equations
# are
# φ_τ(uᵢ) - uᵢ₊₁ = 0 for i < m and S(s) φ_τ(u_m) - u₁ = 0, which closes the orbit, with S(s) the
# shift of a relative periodic orbit, the identity for a periodic one. The equation of
# segment i sits in the row of segment i + 1, that of segment m in row 1. With these reordered
# rows the Jacobian is -I plus a cyclic block subdiagonal of tangent maps. The phase condition
# acts on the Newton step only, so that the scalar part of the residual is zero.
#
# The residual follows the equations: the seeds are advanced over their segments in r, the end of
# the last segment is shifted, and each end point is moved into the row of the seed it has to
# reach, before the seeds are subtracted. r is the only storage.
#
# A search may try a period far from that of its guess, near a point where the Jacobian is nearly
# singular, and a full Newton step may even take the period below zero, where no segment exists.
# The residual of a period out of bounds is infinite, with no integration, so that a search refuses
# the trial as it refuses any step that does not lower the residual. A period that is not positive
# is always out of bounds.

"""
    Residual(G; S=nothing, bounds=(0.0, Inf))

The multiple-shooting residual of the flow `G`, called as `R(r, z)`, as `nkroot!` of
NKRoots.jl calls it. In the row of segment i + 1 it holds the end of segment i minus the seed of
segment i + 1, and in row 1 the end of the last segment, shifted by the shifts of `z`, minus the
first seed,

    rᵢ₊₁ = φ_τ(uᵢ) - uᵢ₊₁,   i = 1, …, m - 1,   r₁ = S(s) φ_τ(u_m) - u₁,   τ = T/m,

and its scalar part is zero. One integration of the seeds per call. Where the period of `z` is
not within `bounds` the residual is infinite, with no integration.

# Arguments
- `G`: the flow, called as `G(X, τ)`.
- `S=nothing`: the shift, called as `S(X, s)`, for relative periodic orbits.
- `bounds::NTuple{2, Real}=(0.0, Inf)`: the least and the largest period, both excluded, as `(0.5T, 2T)` for a
  guess of period `T`. The default refuses the periods that are not positive.

Called as `R(r, z)`, with
- `r::ShootingVector`: overwritten with the residual;
- `z::ShootingVector`: the unknowns, not changed.
"""
struct Residual{GT, ST}
         G::GT                          # the flow
         S::ST                          # the shift, or nothing for periodic orbits
    bounds::Tuple{Float64, Float64}     # the periods outside which the residual is infinite

    Residual(G; S=nothing, bounds::NTuple{2, Real}=(0.0, Inf)) =
        new{typeof(G), typeof(S)}(G,
                                  S,
                                  bounds)
end

function (R::Residual)(r::ShootingVector, z::ShootingVector)
    # ---- input checks ----
    nshifts(z) > 0 && R.S === nothing &&
        throw(ArgumentError("a relative periodic orbit needs its shift S"))

    # ---- a period out of bounds: an infinite residual ----
    R.bounds[1] < period(z) < R.bounds[2] || return r .= Inf

    # ---- the length of a segment, the same for all ----
    #      τ = T/m
    τ = period(z)/nsegments(z)

    # ---- the unknowns: the seeds, and the scalars with them ----
    #      rᵢ = uᵢ,   r.d = z.d
    r .= z

    # ---- advanced over their segments ----
    #      rᵢ = φ_τ(uᵢ)
    R.G(r.X, τ)

    # ---- the end of the last segment shifted ----
    #      r_m = S(s) φ_τ(u_m)
    nshifts(z) > 0 && R.S(tail(r.X), shifts(z))

    # ---- each end point in the row of the seed it has to reach ----
    #      rᵢ₊₁ = φ_τ(uᵢ),   r₁ = S(s) φ_τ(u_m)
    roll_forward!(r.X)

    # ---- minus the unknowns: the seeds, and the scalars, which r still holds from z, so that
    #      the scalar part of the residual is zero ----
    #      rᵢ₊₁ = φ_τ(uᵢ) - uᵢ₊₁,   r₁ = S(s) φ_τ(u_m) - u₁,   r.d = z.d - z.d = 0
    r .= r .- z

    return r
end
