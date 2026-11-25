# Continuation of an orbit in a parameter of the equation.
#
# An orbit at the parameter p lies on a branch, a curve in the unknowns and the parameter
# together, x = (z, p). From a point of the branch with its unit tangent t, the next point is the
# predictor x + h t, a step of length h along the tangent, brought back to the branch by the
# search with the parameter free of parameter.jl, which lands on the branch near the predictor and
# gives the tangent there. The step is halved when the search does not converge in _CORRECTOR
# iterations, or when the tangent makes a cosine below _TURN with the one before, and grows by half
# after a search of _EASY iterations or fewer.
#
# The branch is followed up to an event, where a function of the point changes sign between two
# points: the parameter less its target, or, with no target, the part of the tangent in the
# parameter, which is zero at a fold, where the parameter turns. The point of the event is closed
# in on the step, by the secant rule with the Illinois correction, and the continuation ends there.
# With a target the branch is followed through its folds.

const _CORRECTOR = 8   # iterations of the search from a predictor at most
const _TURN      = 0.7 # least cosine between the tangents of two points
const _EASY      = 3   # iterations of a search up to which the next step grows
const _CLOSE     = 20  # steps tried at most to close in on an event

"""
    continuation!(family, G, L, F, z, p; sense=1, target=nothing, step=0.05, longest=2.0,
                  shortest=1e-4, points=100, kw...) -> (; status, p, path)

Follow the branch of the orbit `z` at the parameter `p` as the parameter changes, by
steps along its tangent, each corrected by the search of [`freesearch!`](@ref). With a `target`
the continuation ends on the orbit at that parameter, through the folds of the branch if any.
With none it ends at the first fold, where the parameter turns and the Jacobian is singular. The
length of a step is that of the change of `(z, p)`.

# Arguments
- `family`, `G`, `L`, `F`: those of [`freesearch!`](@ref).
- `z::ShootingVector`: an orbit at the parameter `p`, overwritten with the last point.
- `p::Real`: its parameter.
- `sense::Int=1`: `1` for a parameter that grows at the start, `-1` for one that shrinks.
- `target=nothing`: the parameter at which the continuation ends, or `nothing` to end at the
  first fold.
- `step::Real=0.05`: the first step.
- `longest::Real=2.0`: the longest step.
- `shortest::Real=1e-4`: the step below which the continuation gives up.
- `points::Int=100`: the most points, the first one included.
- `residual=nothing`: a function `(r, z, p) -> r` in place of the residual of the flow at `p`, for
  a parameter that is not in the dynamics, whose column is then the central difference. With
  `R = Residual(G)` and `R₀` its value at a guess, `(r, z, a) -> (R(r, z); r .-= a .* R₀)` and
  `family = (x, a) -> x` follow the Newton homotopy `R(z) - a R₀ = 0` from `a = 1` to the
  `target` 0.
- `kw...`: the other keywords of [`freesearch!`](@ref), but `maxiter`.

# Returns
A named tuple with
- `status::Symbol`: `:target` or `:fold` at the event, `:points` after `points` points, or `:step`
  when the step falls below `shortest`.
- `p::Float64`: the parameter of the last point.
- `path`: the points, as the vectors `p`, the parameter, `period`, and `slope`, the part of the
  unit tangent in the parameter.
"""
function continuation!(  family,
                              G,
                              L,
                              F,
                              z::ShootingVector,
                              p::Real;
                          sense::Int=1,
                         target=nothing,
                           step::Real=0.05,
                        longest::Real=2.0,
                       shortest::Real=1e-4,
                         points::Int=100,
                                kw...)
    # ---- input checks ----
    sense in (1, -1) ||
        throw(ArgumentError("sense must be 1 or -1"))

    # ---- the tangent of the branch at the orbit, in the sense asked ----
    t  = _extended(z, 0.0)
    t .= 0
    t.d[end] = sense
    out    = _free!(family, G, L, F, z, p, t; kw...)
    p      = out.p
    t      = out.tangent
    h      = float(step)
    status = :points
    path   = (p      = [p],
              period = [period(z)],
              slope  = [t.d[end]])

    # the function of a point whose change of sign is the event, and the share of its value at the
    # last point within which its zero is closed in: the slope comes from GMRES, and at a fold
    # the parameter does not change to first order
    event(o) = target === nothing ? o.tangent.d[end] : o.p - target
    near     = target === nothing ? 1e-2 : 1e-10

    # the point at the step h from the last one: the unknowns and the outcome of the search from
    # the predictor, or nothing when it does not converge, when the tangent turns too far, or when
    # the bases of the preconditioner do not become accurate there
    function point(h)
        zn  = copy(z)
        zn .= z .+ h .* _unknowns(t)
        o   = try
            _free!(family, G, L, F, zn, p + h*t.d[end], t; kw..., maxiter=_CORRECTOR)
        catch e
            e isa ErrorException ||
                rethrow()
            return nothing
        end
        (o.status === :converged && LinearAlgebra.dot(o.tangent, t) >= _TURN) ||
            return nothing

        return zn, o
    end

    while length(path.p) < points
        # ---- the next point, the step halved until there is one ----
        next = point(h)
        while next === nothing && h/2 >= shortest
            h   /= 2
            next = point(h)
        end
        next === nothing &&
            (status = :step; break)

        # ---- an event between the two points: its zero is closed in on the step, between the
        #      last point, a, and the new one, b, by the secant rule, with the value at an end
        #      that stays halved, and the continuation ends on the closest point found ----
        g = event((p = p, tangent = t))
        if g*event(next[2]) < 0
            status = target === nothing ? :fold : :target
            a      = (h = 0.0, g = g)
            b      = (h = h, g = event(next[2]))
            side   = 0
            for _ in 1:_CLOSE
                hs    = (a.h*b.g - b.h*a.g)/(b.g - a.g)
                trial = point(hs)
                trial === nothing &&
                    break
                next = trial
                gs   = event(trial[2])
                abs(gs) <= near*abs(g) &&
                    break
                if gs*a.g < 0
                    b    = (h = hs, g = gs)
                    side == -1 && (a = (h = a.h, g = a.g/2))
                    side = -1
                else
                    a    = (h = hs, g = gs)
                    side == 1 && (b = (h = b.h, g = b.g/2))
                    side = 1
                end
            end
        end

        # ---- the point is taken, and the next step grows after an easy search ----
        z .= next[1]
        p  = next[2].p
        t  = next[2].tangent
        push!(path.p, p)
        push!(path.period, period(z))
        push!(path.slope, t.d[end])
        status === :points ||
            break
        next[2].iter <= _EASY && (h = min(1.5h, longest))
    end

    return (status = status,
            p      = p,
            path   = path)
end
