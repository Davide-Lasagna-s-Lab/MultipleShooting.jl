# The interface: what multiple shooting needs from the caller. Two things, both of types of the
# caller's choice, of which the package assumes nothing: the seeds, and the propagator. This file
# declares their functions, with no method but the default of dpropagate!. Those of the seeds have
# methods for two kinds of them, an array of numbers, a Batched (batched.jl), and a vector of
# states, a Sliced (sliced.jl); for seeds of another kind, and for the propagator, the caller
# writes them.
#
# The seeds. X holds states, m of them: the seeds of the segments, or vectors attached to them,
# one per seed. What X is, an array on a CPU or on a GPU, a vector of states, a type of the
# caller's, the rest of the package does not know. It only moves the states about, by
#
#     nsegments(X)            the number of states of X
#     states(X, r)            the states r of X, as an X that shares them
#     head(X), tail(X)        the first and the last state, from states
#     similar_states(X, b)    an X with b states, its entries not set
#     roll_forward!(X)        the states moved forward by one place, cyclically
#     roll_backward!(X)       and back by one
#     dots(X, Y)              the inner products of the states of X and Y, one per state
#     coefficients(X, c)      the coefficients c, one per state, as dots gives them
#
# and combines them, by `similar`, `copy`, `copyto!`, `randn!` and broadcasting, of Base.
#
# The propagator. P advances all the states of an X at once, on threads or on a GPU, or one at a
# time, with the methods
#
#     propagate!(P, X, τ)      advance every state of X by the time τ          residual
#     dpropagate!(P, X, V, τ)  as propagate!, and V the derivative in τ of the  period
#                              end points; by default f at the end points
#     rhs!(P, F, X)            the time derivative f at every state of X       period, phase
#     tangent!(P, X, V, τ)     advance the states of X, and those of V by      Jacobian, subspace
#                              the tangent equation along them; V like X, or
#                              a vector of k′ of them
#     shift!(P, X, s)          shift every state of X by the shifts s          relative orbits
#     dshift!(P, G, X, j)      the generator of shift j at every state of X    relative orbits
#     forced!(P, X, V, τ)      as tangent!, forced by ∂f/∂p                    sensitivity
#
# shift! and dshift! are needed for relative periodic orbits only, and forced! for the sensitivity
# to a parameter only. dpropagate! has a default, the right-hand side at the end points, the
# derivative of the exact flow; a propagator gives its own for the derivative of its integrator.
# For relative periodic orbits the system must be equivariant: the shifts commute with the flow,
# and f(S(s) x) = S(s) f(x).


# ---------------------------------------------------------------------------- #
# the seeds                                                                    #
# ---------------------------------------------------------------------------- #

"""
    nsegments(X) -> Int

The number of states of `X`: for the seeds of an orbit, the number of its segments.
"""
function nsegments end

"""
    states(X, r) -> Y

The states `r`, a range, of `X`, as an object like `X` that shares them, without a copy: the
first seed, `states(X, 1:1)`, or the end of the last segment.
"""
function states end

"""
    head(X) -> Y
    tail(X) -> Y

The first and the last state of `X`, as an object like `X` of one state that shares it: the first
seed, and the end of the last segment. From [`states`](@ref), for any `X`.
"""
head(X) = states(X, 1:1)
tail(X) = states(X, nsegments(X):nsegments(X))

"""
    similar_states(X, b) -> Y

An object like `X` with `b` states, its entries not set, which the propagator can advance.
"""
function similar_states end

"""
    roll_forward!(X) -> X

Move the states of `X` forward by one place, forward in time along the orbit, cyclically and in
place: state i goes into place i + 1, and the last state into place 1,

    X = (x₁, x₂, x₃)   →   X = (x₃, x₁, x₂)

In multiple shooting this puts the end of each segment in the row of the seed it has to reach.
"""
function roll_forward! end

"""
    roll_backward!(X) -> X

Move the states of `X` back by one place, backward in time, cyclically and in place, the inverse
of [`roll_forward!`](@ref): state i goes into place i - 1, and the first state into the last
place,

    X = (x₁, x₂, x₃)   →   X = (x₂, x₃, x₁)
"""
function roll_backward! end

"""
    dots(X, Y) -> c

The inner products of the states of `X` and `Y` one by one, real, in a form that broadcasts
against an object like `X`, so that `c .* X` is every state times its own product. GMRES, the phase
conditions, the bases of the subspace iteration and the preconditioner all use it, so that they
share one geometry.
"""
function dots end

"""
    coefficients(X, c) -> a

The coefficients of the vector `c`, one per state of `X`, in the form of [`dots`](@ref), so that
`a .* X` is every state times its own.
"""
function coefficients end


# ---------------------------------------------------------------------------- #
# the propagator                                                               #
# ---------------------------------------------------------------------------- #

"""
    propagate!(P, X, τ) -> X

Advance every state of `X` in place by the time `τ`, with the integrator of the propagator,

    xᵢ ← φ_τ(xᵢ),   i = 1, …, m,

with φ_τ the flow map of x' = f(x), as the integrator computes it.

# Arguments
- `P`: the propagator, the type the method is written for.
- `X`: the states, overwritten with their images.
- `τ::Real`: the time.
"""
function propagate! end

"""
    dpropagate!(P, X, V, τ) -> V

Advance every state of `X` by the time `τ`, as [`propagate!`](@ref), and write into `V` the
derivative of the end points in `τ`, the column of the period of the [`Jacobian`](@ref),

    xᵢ ← φ_τ(xᵢ),   vᵢ ← ∂φ_τ(xᵢ)/∂τ.

The default is the right-hand side at the end points, the derivative of the exact flow,

    vᵢ ← f(φ_τ(xᵢ)),

which differs from that of the integrator by the error of the scheme. A propagator gives its own
method for the derivative of its integrator. When the steps see `τ` only through the products of
their length with the right-hand side, as those of a Runge–Kutta scheme with a fixed number of
steps do, it is the tangent equation from zero forced by the right-hand side over the time,

    v' = ∂f/∂x(x) v + f(x)/τ,   v(0) = 0,   vᵢ ← v(τ),

integrated by the stages of the scheme, each forced at its own point.

# Arguments
- `P`: the propagator.
- `X`: the states, overwritten with their images.
- `V`: like `X`, overwritten with the derivatives of the end points in `τ`.
- `τ::Real`: the time.
"""
dpropagate!(P, X, V, τ::Real) = (propagate!(P, X, τ); rhs!(P, V, X); V)

"""
    rhs!(P, F, X) -> F

The time derivative of the system x' = f(x) at every state of `X`, written into `F`,

    Fᵢ = f(xᵢ).

# Arguments
- `P`: the propagator.
- `F`: like `X`, overwritten with the time derivatives.
- `X`: the states, not changed.
"""
function rhs! end

"""
    tangent!(P, X, V, τ) -> V

Advance every state of `X` by the time `τ`, and the perturbations `V` by the tangent equation
along them, all in place. `V` is like `X`, one perturbation of every state, for a product of the
Jacobian, or a vector of k′ objects like `X`, `V[a]` the perturbation a of every state, for the k′
directions of every shooting point of a subspace step: a propagator gives a method for each. All
are advanced along the states of `X`, which are advanced once,

    xᵢ ← φ_τ(xᵢ),   vᵢᵃ ← ψᵢ vᵢᵃ,   ψᵢ = ∂φ_τ/∂x(xᵢ),   a = 1, …, k′,

with ψᵢ v the solution at τ of v' = ∂f/∂x(x(t)) v from v(0) = v, along the trajectory x(t) from
xᵢ: the tangent map of the integrator, for the Jacobian of its residual to be exact. How the k′
directions are advanced together, at once on a GPU or one after the other, is for the propagator.

# Arguments
- `P`: the propagator.
- `X`: the m states, overwritten with their images.
- `V`: m perturbations, like `X`, or a vector of k′ of them, overwritten with their images by the
  tangent maps.
- `τ::Real`: the time.
"""
function tangent! end

"""
    shift!(P, X, s) -> X

Shift every state of `X` in place, by the continuous symmetries of the system with the shifts
s = (s₁, …, s_c),

    xᵢ ← S(s) xᵢ,

as u(x) ← u(x + s) for a translation of a periodic domain. Relative periodic orbits only.

# Arguments
- `P`: the propagator.
- `X`: the states, overwritten with the shifted states.
- `s`: the shifts, a vector with one per continuous symmetry.
"""
function shift! end

"""
    dshift!(P, G, X, j) -> G

The generator of the `j`-th shift at every state of `X`, the derivative of the shifted state with
respect to `sⱼ` at `sⱼ = 0`, written into `G`,

    Gᵢ = gⱼ(xᵢ) = ∂(S(s) xᵢ)/∂sⱼ  at s = 0,

as gⱼ(u) = uₓ for a translation. Relative periodic orbits only.

# Arguments
- `P`: the propagator.
- `G`: like `X`, overwritten with the generators.
- `X`: the states, not changed.
- `j::Int`: the index of the shift.
"""
function dshift! end

"""
    forced!(P, X, V, τ) -> V

As [`tangent!`](@ref), with the tangent equation forced by the derivative of the right-hand side
in the parameter p of the equation,

    v' = ∂f/∂x(x) v + ∂f/∂p(x),   v(0) = vᵢ,   vᵢ ← v(τ).

From `V = 0` it gives the derivative of the end points in the parameter, vᵢ = ∂φ_τ(xᵢ)/∂p. For
[`tangent_sensitivity`](@ref) only.

# Arguments
- `P`: the propagator, at the value of the parameter.
- `X`: the states, overwritten with their images.
- `V`: like `X`, overwritten with the solutions of the forced tangent equation.
- `τ::Real`: the time.
"""
function forced! end


# ---------------------------------------------------------------------------- #
# internal: built on the interface of the seeds                                #
# ---------------------------------------------------------------------------- #

# the inner product of X and Y, the sum over their states
_dot(X, Y) = sum(dots(X, Y))
