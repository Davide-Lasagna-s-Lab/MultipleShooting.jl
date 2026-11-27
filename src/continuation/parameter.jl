# A parameter of the equation as one more unknown.
#
# The objects of the dynamics depend on a parameter p through a function family(x, p) of the
# caller's: the object x at the parameter p. The search moves by it every object it holds: the
# flow G of the residual R, the tangent L, the right-hand side F, the shift S and its generator
# dS, the derivative dG, the forced tangent Lp, and those of the preconditioner. The residual is
# then R(z, p), and its zeros are not isolated: each orbit lies on a branch, a curve in the
# unknowns and the parameter together, x = (z, p). The linearised equations have one more
# unknown than equations,
#
#     J δz + c δp = -R,   c = ∂R/∂p,
#
# with J the Jacobian of R at the parameter p and c the column of the parameter: the derivative of
# the end points, from the tangent equation forced by ∂f/∂p, Lp, as for the sensitivity
# (sensitivity/sensitivity.jl), or, with no Lp, a central difference of R in p. The solutions
# form a line along the null vector n of [J c], which at an orbit is the tangent of its branch.
# The search takes the shortest one, at right angles to n, the Newton step with the Moore–Penrose
# inverse of [J c] (Allgower & Georg 1990), the solution of the bordered system
#
#     [ J    c  ] [ δz ]     [ R ]
#     [ n_zᵀ n_p ] [ δp ] = - [ 0 ],
#
# and n solves the same matrix with the right-hand side (0, 1) and, in the last row, the n of the
# iterate before, which fixes its sense. Where J is singular, at a fold of the branch, the
# bordered matrix is not, so that the search converges at a fold as anywhere else on the branch
# (the bordering of Keller 1977; Newton–Krylov for orbits, Sánchez et al. 2004, Sánchez & Net
# 2010). It lands on the branch near the guess, at the parameter where it meets it, and does not
# aim at a parameter: for a guess near an orbit that exists at another parameter only, a ghost,
# where a search at a fixed parameter stalls, it converges.
#
# Both systems are solved by GMRES (gmres.jl), with the preconditioner of J on the right and the
# identity on the parameter. The step is shortened by backtracking, halved until the squared
# residual falls by a share _ETA of the fall of a linear problem, μ(2 - μ) for the step μ δ. An
# extended vector is a ShootingVector with the parameter as one more scalar, after the period and
# the shifts; its norm weighs p like any other scalar.

const _ETA = 1e-3 # share of the fall of a linear problem that a step must reach

# the unknowns z, or a vector like them, and the parameter p as one vector, with seeds of its own
_extended(z::ShootingVector, p::Real) = ShootingVector(copy(z.X), [z.d; p]; w=z.w)

# the unknowns of an extended vector, sharing its seeds
_unknowns(x::ShootingVector) = ShootingVector(x.X, x.d[1:(end - 1)]; w=x.w)

# the preconditioner M of the unknowns applied to an extended vector, the identity on the
# parameter, or no preconditioner at all
_precondition(::Nothing, y::ShootingVector) = y

function _precondition(M, y::ShootingVector)
    v = _unknowns(y)
    o = LinearAlgebra.ldiv!(similar(v), M, v)

    return ShootingVector(o.X, [o.d; y.d[end]]; w=y.w)
end

# the object x at the parameter p by the family, or nothing for none
_at(family, x, p::Real) = x === nothing ? nothing : family(x, p)

# the Jacobian J, and the preconditioner M when there is one, moved to the unknowns z and to the
# parameter p, each of their objects moved there by the family
function _move!(J::Jacobian, M, family, p::Real, z::ShootingVector)
    # ---- the Jacobian ----
    J.L  = family(J.L, p)
    J.F  = family(J.F, p)
    J.S  = _at(family, J.S, p)
    J.dS = _at(family, J.dS, p)
    J.dG = _at(family, J.dG, p)
    NKRoots.update!(J, z)

    M === nothing &&
        return J

    # ---- the preconditioner, and its subspace ----
    M.S.L = family(M.S.L, p)
    M.S.S = _at(family, M.S.S, p)
    M.F   = family(M.F, p)
    M.dS  = _at(family, M.dS, p)
    NKRoots.update!(M, z)

    return J
end

# the column of the parameter at (z, p), c = ∂R/∂p, by the central difference of the residual R!
# (r, z, p) -> r, with the step ϵ: two integrations of the seeds
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

# the search with the parameter free, from the unknowns z, changed to the last iterate, and the
# parameter p, with the flow G, the tangent L and the right-hand side F, and the extended vector t
# in the last row of the first system: how it ends, its iterations, the last residual norm and
# parameter, and the unit tangent. The tangent is computed at each iterate, for its step; with
# `tangent = true` also at the last one, which costs the Jacobian there, the column and one
# bordered solve more, and which continuation! needs for its next predictor. With `false` the
# tangent returned is that of the iterate before the last
function _free!(    family,
                         G,
                         L,
                         F,
                         z::ShootingVector,
                         p::Real,
                         t::ShootingVector;
                        Lp=nothing,
                         S=nothing,
                        dS=nothing,
                        dG=nothing,
                   precond=nothing,
                  residual=nothing,
                       tol::Real=1e-10,
                   maxiter::Int=100,
                    bounds::NTuple{2, Real}=(-Inf, Inf),
                   periods::NTuple{2, Real}=(0.0, Inf),
                         ϵ::Real=1e-6,
                krylov_tol::Real=1e-3,
                krylov_dim::Int=200,
                   tangent::Bool=true)
    # ---- the residual R at the parameter q, by the flow and the shift there, or the one given ----
    R(q) = Residual(family(G, q); S=_at(family, S, q), bounds=periods)
    F!   = residual === nothing ? (r, u, q) -> R(q)(r, u) : residual

    # ---- the column of the parameter: by the forced tangent, or the difference of the
    #      residual ----
    forced = Lp !== nothing && residual === nothing

    J      = Jacobian(family(L, p), family(F, p), z; S  = _at(family, S, p),
                                                     dS = _at(family, dS, p),
                                                     dG = _at(family, dG, p))
    r      = F!(similar(z), z, p)
    zn     = similar(z)
    rn     = similar(z)
    res    = LinearAlgebra.norm(r)
    status = :maxiter
    iter   = 0

    while true
        # ---- the last iterate, with no tangent asked there ----
        (res < tol || iter == maxiter) && !tangent &&
            (status = res < tol ? :converged : :maxiter; break)

        # ---- the Jacobian, the preconditioner and the column at the iterate, and the tangent n,
        #      the sense of which is fixed by the one before ----
        _move!(J, precond, family, p, z)
        c = forced ? _forced(family(Lp, p), J.S, z) : _difference(F!, z, p, ϵ)
        t = _tangent(J, precond, c, t; tol=krylov_tol, maxiter=krylov_dim)
        res < tol &&
            (status = :converged; break)
        iter == maxiter &&
            break

        # ---- the shortest step, the solution at right angles to the tangent:
        #      [J c; nᵀ] δ = -(R, 0) ----
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
    freesearch!(family, G, L, F, z, p; Lp=nothing, S=nothing, dS=nothing, dG=nothing,
                precond=nothing, residual=nothing, tol=1e-10, maxiter=100, bounds=(-Inf, Inf),
                periods=(0.0, Inf), ϵ=1e-6, krylov_tol=1e-3, krylov_dim=200)
        -> (; status, iter, res, p)

The search for an orbit with the parameter of the equation as one more unknown, from the guess
`z` at the parameter `p`. With R the [`Residual`](@ref) at the parameter and J its
[`Jacobian`](@ref), each iteration takes the shortest solution of the linearised equations,

    J δz + c δp = -R,   c = ∂R/∂p,   ‖δz‖² + δp² least,

as the solution of the bordered system

    [ J    c  ] [ δz ]     [ R ]
    [ n_zᵀ n_p ] [ δp ] = - [ 0 ],

with n = (n_z, n_p) the unit null vector of [J c], the tangent of the branch at the iterate, and
halves the step until the squared residual falls by a share of the fall of a linear problem. The
bordered matrix is regular at a fold, where J is not. The search ends on the branch near the
guess, at the parameter where it meets it, and does not aim at a parameter: a guess near an orbit
that exists at another parameter only, where the search at a fixed parameter stalls, converges
there. The norm of a step weighs `δp` like the period or a shift of `z`: scale `p` in `family`
when its size differs.

# Arguments
- `family`: a function `(x, p) -> x`, the object `x` at the parameter `p`, for each object of the
  search (`G`, `L`, `F`, `S`, `dS`, `dG`, `Lp` and those of `precond`). It is called with the
  object at any parameter and must give the one at `p` from `p` alone. An object that does not
  depend on the parameter is returned as it is.
- `G`: the flow of the [`Residual`](@ref), at any parameter.
- `L`, `F`: the tangent and the right-hand side of the [`Jacobian`](@ref), at any parameter.
- `z::ShootingVector`: the guess, overwritten with the last iterate.
- `p::Real`: the parameter of the guess.
- `Lp=nothing`: the forced tangent, as for [`tangent_sensitivity`](@ref), which gives the column
  `c` in one integration of the seeds, for a parameter that the shifts do not depend on. With
  none, `c` is the central difference of R in `p`, two integrations, whatever the parameter
  enters.
- `S=nothing`, `dS=nothing`, `dG=nothing`: those of the [`Jacobian`](@ref).
- `precond=nothing`: a [`Preconditioner`](@ref) built at `z` and `p` with its own tangent and
  right-hand side, moved with the iterates and the parameter, or none.
- `residual=nothing`: a function `(r, z, p) -> r` in place of R, see [`continuation!`](@ref);
  `c` is then the central difference.
- `tol::Real=1e-10`: the residual norm of a converged search.
- `maxiter::Int=100`: the most iterations.
- `bounds=(-Inf, Inf)`: the least and the largest parameter, both excluded. A trial step outside
  is refused with no integration.
- `periods=(0.0, Inf)`: the same for the period, as the `bounds` of [`Residual`](@ref).
- `ϵ::Real=1e-6`: the step of the central difference in `p`, with no `Lp`.
- `krylov_tol::Real=1e-3`: the relative residual of GMRES.
- `krylov_dim::Int=200`: the most iterations of GMRES.

# Returns
A named tuple with
- `status::Symbol`: `:converged`, `:maxiter`, or `:collapsed` when no step is accepted.
- `iter::Int`: the iterations made.
- `res::Float64`: the last residual norm.
- `p::Float64`: the last parameter.
"""
function freesearch!(family, G, L, F, z::ShootingVector, p::Real; kw...)
    # ---- the parameter alone in the last row of the first system ----
    t  = _extended(z, 0.0)
    t .= 0
    t.d[end] = 1

    out = _free!(family, G, L, F, z, p, t; kw..., tangent=false)

    return (status = out.status,
            iter   = out.iter,
            res    = out.res,
            p      = out.p)
end
