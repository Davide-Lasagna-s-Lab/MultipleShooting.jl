# A parameter of the equation as one more unknown.
#
# The propagator depends on a parameter p, through a function p -> P that the caller gives. The
# residual is then F(z, p), and its zeros are no longer isolated: each orbit lies on a branch, a
# curve in the unknowns and the parameter together, x = (z, p). The linearised equations have one
# more unknown than equations,
#
#     J δz + c δp = -F,   c = ∂F/∂p,
#
# with J the Jacobian at the parameter p and c the column of the parameter, from the tangent
# equation forced by the derivative of the right-hand side in p, as for the sensitivity
# (sensitivity/sensitivity.jl), or from a central difference of the residual in p. Their solutions form a
# line, along the null vector n of [J c],
# which at an orbit is the tangent of its branch. The search takes the shortest one, at right
# angles to n, the solution of the bordered system
#
#     [ J    c  ] [ δz ]     [ F ]
#     [ n_zᵀ n_p ] [ δp ] = - [ 0 ],
#
# and n solves the same system with the right-hand side (0, 1) and the n of the iterate before in
# the last row, which also keeps its sense. Where J is singular, at a fold of the branch, the
# bordered matrix is not, so that the search converges at a fold as anywhere else on the branch.
# It lands on the branch near the guess, at the parameter where it meets it, and does not aim at a
# parameter: for a guess near an orbit that exists at another parameter only, a ghost, this is
# where a search at a fixed parameter stalls.
#
# Both systems are solved by GMRES (gmres.jl), with the preconditioner of J on the right and the
# identity on the parameter. The step is shortened by backtracking, halved until the residual falls
# by a share of what a linear problem would give. An extended vector is a ShootingVector with the
# parameter as one more scalar, after the period and the shifts.

const _ETA = 1e-3 # share of the fall of a linear problem that a step must reach

# the unknowns z, or a vector like them, and the parameter p as one vector, with seeds of its own
_extended(z::ShootingVector, p::Real) = ShootingVector(copy(z.X), [z.d; p])

# the unknowns of an extended vector, sharing its seeds
_unknowns(x::ShootingVector) = ShootingVector(x.X, x.d[1:(end - 1)])

# the preconditioner M of the unknowns applied to an extended vector, the identity on the
# parameter, or no preconditioner at all
_precondition(::Nothing, y::ShootingVector) = y

function _precondition(M, y::ShootingVector)
    v = _unknowns(y)
    o = LinearAlgebra.ldiv!(similar(v), M, v)

    return ShootingVector(o.X, [o.d; y.d[end]])
end

# the Jacobian J, and the preconditioner M when there is one, moved to the unknowns z with the
# propagator P of a new parameter
function _move!(J::Jacobian, M, P, z::ShootingVector)
    J.P = P
    NKRoots.update!(J, z)
    M === nothing &&
        return J
    M.S.P = P
    NKRoots.update!(M, z)

    return J
end

# the column of the parameter at (z, p), c = ∂F/∂p, by the central difference of the residual F!
# with the step ϵ, two integrations of the seeds
function _difference(F!, z::ShootingVector, p::Real, ϵ::Real)
    c  = F!(similar(z), z, p + ϵ)
    b  = F!(similar(z), z, p - ϵ)
    c .= (c .- b) ./ (2ϵ)

    return c
end

# the solution of the bordered system with the Jacobian J, the column c, the row t and the
# right-hand side b, the last two extended vectors,
#
#     [ J    c  ] [ δz ]   [ b_z ]
#     [ t_zᵀ t_p ] [ δp ] = [ b_p ],
#
# by GMRES with the preconditioner M on the right
function _bordered(J::Jacobian, M, c::ShootingVector,
                                   t::ShootingVector,
                                   b::ShootingVector; tol, maxiter)
    function A!(out, y)
        v  = _precondition(M, y)
        o  = J*_unknowns(v)
        o .= o .+ v.d[end] .* c
        copyto!(out.X, o.X)
        out.d[1:(end - 1)] .= o.d
        out.d[end]          = LinearAlgebra.dot(t, v)

        return out
    end

    return _precondition(M, gmres(A!, b; tol=tol, maxiter=maxiter).x)
end

# the unit null vector of [J c], with a positive product with the extended vector t: the solution
# of the bordered system with t in the last row and the right-hand side (0, 1)
function _tangent(J::Jacobian, M, c::ShootingVector, t::ShootingVector; tol, maxiter)
    b  = similar(t)
    b .= 0
    b.d[end] = 1
    n  = _bordered(J, M, c, t, b; tol=tol, maxiter=maxiter)
    n .= n ./ LinearAlgebra.norm(n)

    return n
end

# the search with the parameter free from the unknowns z, changed to the last iterate, and the
# parameter p, with the extended vector t in the last row of the first system: how it ends, its
# iterations, the last residual and parameter, and the unit tangent at the last iterate
function _free!(family, z::ShootingVector, p::Real, t::ShootingVector;   precond=nothing,
                                                                        residual=nothing,
                                                                          column::Symbol=:forced,
                                                                             tol::Real=1e-10,
                                                                         maxiter::Int=100,
                                                                          bounds::NTuple{2, Real}=(-Inf, Inf),
                                                                         periods::NTuple{2, Real}=(0.0, Inf),
                                                                               ϵ::Real=1e-6,
                                                                      krylov_tol::Real=1e-3,
                                                                      krylov_dim::Int=200)
    # ---- input checks ----
    column in (:forced, :difference) ||
        throw(ArgumentError("column must be :forced or :difference"))

    F!     = residual === nothing ? (r, u, q) -> Residual(family(q); bounds=periods)(r, u) : residual
    J      = Jacobian(family(p), z)
    r      = F!(similar(z), z, p)
    zn     = similar(z)
    rn     = similar(z)
    res    = LinearAlgebra.norm(r)
    status = :maxiter
    iter   = 0

    while true
        # ---- the Jacobian, the preconditioner and the column at the iterate, and the tangent ----
        _move!(J, precond, family(p), z)
        c = column === :forced && residual === nothing ? _forced(J.P, z) : _difference(F!, z, p, ϵ)
        t = _tangent(J, precond, c, t; tol=krylov_tol, maxiter=krylov_dim)
        res < tol &&
            (status = :converged; break)
        iter == maxiter &&
            break

        # ---- the shortest step, the solution at right angles to the tangent ----
        b  = _extended(r, 0.0)
        b .= .-b
        δ  = _bordered(J, precond, c, t, b; tol=krylov_tol, maxiter=krylov_dim)

        # ---- backtracking: the step halved until the squared residual falls by the share _ETA
        #      of the fall of a linear problem, or the step is lost in the rounding of z. A trial
        #      parameter out of bounds is refused with no integration ----
        μ        = 1.0
        pn       = p
        resn     = Inf
        accepted = false
        while μ*LinearAlgebra.norm(δ) > eps()*(1 + LinearAlgebra.norm(z))
            zn      .= z .+ μ .* _unknowns(δ)
            pn       = p + μ*δ.d[end]
            resn     = bounds[1] < pn < bounds[2] ? LinearAlgebra.norm(F!(rn, zn, pn)) : Inf
            accepted = 1 - (resn/res)^2 > _ETA*μ*(2 - μ)
            accepted &&
                break
            μ /= 2
        end
        accepted ||
            (status = :collapsed; break)

        # ---- the new iterate ----
        z   .= zn
        r   .= rn
        p    = pn
        res  = resn
        iter += 1
    end

    return (status  = status,
            iter    = iter,
            res     = res,
            p       = p,
            tangent = t)
end

"""
    freesearch!(family, z, p; precond=nothing, column=:forced, tol=1e-10, maxiter=100,
                bounds=(-Inf, Inf), periods=(0.0, Inf), ϵ=1e-6, krylov_tol=1e-3, krylov_dim=200)
                -> (; status, iter, res, p)

The search for an orbit with a parameter of the equation as one more unknown, from the guess `z`
at the parameter `p`. Each iteration takes the shortest solution of the linearised equations,

    J δz + c δp = -F,   c = ∂F/∂p,   ‖δz‖² + δp² least,

with `J` the Jacobian at the propagator `family(p)` and `c` the column of the parameter, and
halves it until the residual falls. The search ends on an orbit of `family(p)` at the
parameter `p` where it lands on a branch, and does not aim at a parameter. A guess near an orbit
that exists at another parameter only, where the search at a fixed parameter stalls, converges
there. The step counts a change of the parameter as a change of the unknowns of the same size: a
parameter is weighed by its scale in `family`.

# Arguments
- `family`: a function `p -> P`, the propagator at the parameter `p`, of one type for all `p`.
- `z::ShootingVector`: the guess, overwritten with the last iterate.
- `p::Real`: the parameter of the guess.
- `precond=nothing`: a [`Preconditioner`](@ref) built at `z` with `family(p)`, moved with the
  iterates, or none.
- `column::Symbol=:forced`: `:forced` takes the column from the tangent equation forced by the
  derivative of the right-hand side in the parameter, [`forced!`](@ref) of the propagator, one
  integration of the seeds, for a parameter that the shifts do not depend on. `:difference`
  takes the central difference of the residual in `p`, two integrations, whatever the parameter
  enters.
- `tol::Real=1e-10`: the residual of a converged search.
- `maxiter::Int=100`: the most iterations.
- `bounds=(-Inf, Inf)`: the least and the largest parameter, both excluded. A trial step outside
  is refused.
- `periods=(0.0, Inf)`: the same for the period, as the `bounds` of [`Residual`](@ref).
- `ϵ::Real=1e-6`: the step of the central difference in `p`, for `column=:difference`.
- `krylov_tol::Real=1e-3`: the relative residual of GMRES.
- `krylov_dim::Int=200`: the most iterations of GMRES.

# Returns
A named tuple with
- `status::Symbol`: `:converged`, `:maxiter`, or `:collapsed` when no step is accepted.
- `iter::Int`: the iterations made.
- `res::Float64`: the last residual.
- `p::Float64`: the last parameter.
"""
function freesearch!(family, z::ShootingVector, p::Real; kw...)
    # ---- the parameter alone in the last row of the first system ----
    t  = _extended(z, 0.0)
    t .= 0
    t.d[end] = 1

    out = _free!(family, z, p, t; kw...)

    return (status = out.status,
            iter   = out.iter,
            res    = out.res,
            p      = out.p)
end
