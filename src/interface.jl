# The propagator: what multiple shooting needs from the dynamical system. It acts on batches of
# states, the seeds of the segments or the vectors attached to them (see seeds.jl), so that it can
# advance all the states of a batch at once, on threads or on a GPU, or one at a time. The package
# passes it the batches of the caller as they are, and those it makes from them. A propagator is a
# type of the user's, for which these methods are defined:
#
#     propagate!(P, X, τ)      advance every state of X by the time τ          residual
#     dpropagate!(P, X, V, τ)  as propagate!, and V the derivative in τ of the  period
#                              end points; by default f at the end points
#     rhs!(P, F, X)            the time derivative f at every state of X       period, phase
#     tangent!(P, X, V, τ)     advance the states of X, and those of V by      Jacobian, subspace
#                              the tangent equation along them; V a batch
#                              like X, or a vector of k′ of them
#     shift!(P, X, s)          shift every state of X by the shifts s          relative orbits
#     dshift!(P, G, X, j)      the generator of shift j at every state of X    relative orbits
#     forced!(P, X, V, τ)      as tangent!, forced by ∂f/∂p                    sensitivity
#
# shift! and dshift! are needed for relative periodic orbits only, and forced! for the sensitivity
# to a parameter only. dpropagate! has a default, the right-hand side at the end points, the
# derivative of the exact flow; a propagator gives its own for the derivative of its integrator.
# For relative periodic orbits the system must be equivariant: the shifts commute with the flow,
# and f(S(s) x) = S(s) f(x).

"""
    propagate!(P, X, τ) -> X

Advance every state of the batch `X` in place by the time `τ`, with the integrator of the
propagator,

    xᵢ ← φ_τ(xᵢ),   i = 1, …, m,

with φ_τ the flow map of x' = f(x), as the integrator computes it.

# Arguments
- `P`: the propagator, the type the method is written for.
- `X`: a batch of states, overwritten with their images.
- `τ::Real`: the time.
"""
function propagate! end

"""
    dpropagate!(P, X, V, τ) -> V

Advance every state of the batch `X` by the time `τ`, as [`propagate!`](@ref), and write into `V`
the derivative of the end points in `τ`, the column of the period of the [`Jacobian`](@ref),

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
- `X`: a batch of states, overwritten with their images.
- `V`: a batch like `X`, overwritten with the derivatives of the end points in `τ`.
- `τ::Real`: the time.
"""
dpropagate!(P, X, V, τ::Real) = (propagate!(P, X, τ); rhs!(P, V, X); V)

"""
    rhs!(P, F, X) -> F

The time derivative of the system x' = f(x) at every state of the batch `X`, written into `F`,

    Fᵢ = f(xᵢ).

# Arguments
- `P`: the propagator.
- `F`: a batch like `X`, overwritten with the time derivatives.
- `X`: a batch of states, not changed.
"""
function rhs! end

"""
    tangent!(P, X, V, τ) -> V

Advance every state of the batch `X` by the time `τ`, and the perturbations `V` by the tangent
equation along them, all in place. `V` is a batch like `X`, one perturbation of every state, for a
product of the Jacobian, or a vector of k′ such batches, `V[a]` the perturbation a of every state,
for the k′ directions of every shooting point of a subspace step: a propagator gives a method for
each. Every batch is advanced along the states of `X`, which are advanced once,

    xᵢ ← φ_τ(xᵢ),   vᵢᵃ ← ψᵢ vᵢᵃ,   ψᵢ = ∂φ_τ/∂x(xᵢ),   a = 1, …, k′,

with ψᵢ v the solution at τ of v' = ∂f/∂x(x(t)) v from v(0) = v, along the trajectory x(t) from
xᵢ: the tangent map of the integrator, for the Jacobian of its residual to be exact. How the k′
batches are advanced together, at once on a GPU or one after the other, is for the propagator.

# Arguments
- `P`: the propagator.
- `X`: a batch of m states, overwritten with their images.
- `V`: a batch of m perturbations, or a vector of k′ of them, overwritten with their images by the
  tangent maps.
- `τ::Real`: the time.
"""
function tangent! end

"""
    shift!(P, X, s) -> X

Shift every state of the batch `X` in place, by the continuous symmetries of the system with the
shifts s = (s₁, …, s_c),

    xᵢ ← S(s) xᵢ,

as u(x) ← u(x + s) for a translation of a periodic domain. Relative periodic orbits only.

# Arguments
- `P`: the propagator.
- `X`: a batch of states, overwritten with the shifted states.
- `s`: the shifts, a vector with one per continuous symmetry.
"""
function shift! end

"""
    dshift!(P, G, X, j) -> G

The generator of the `j`-th shift at every state of the batch `X`, the derivative of the shifted
state with respect to `sⱼ` at `sⱼ = 0`, written into `G`,

    Gᵢ = gⱼ(xᵢ) = ∂(S(s) xᵢ)/∂sⱼ  at s = 0,

as gⱼ(u) = uₓ for a translation. Relative periodic orbits only.

# Arguments
- `P`: the propagator.
- `G`: a batch like `X`, overwritten with the generators.
- `X`: a batch of states, not changed.
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
- `X`: a batch of states, overwritten with their images.
- `V`: a batch like `X`, overwritten with the solutions of the forced tangent equation.
- `τ::Real`: the time.
"""
function forced! end
