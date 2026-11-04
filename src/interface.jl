# The propagator: what multiple shooting needs from the dynamical system. It acts on batches of
# states, the seeds of the segments or the vectors attached to them (see batch.jl), so that it can
# advance all the states of a batch at once, on threads or on a GPU, or one at a time. A
# propagator is a type of the user's, for which these methods are defined:
#
#     propagate!(P, X, τ)      advance every state of X by the time τ          residual
#     rhs!(P, F, X)            the time derivative f at every state of X       period, phase
#     tangent!(P, X, V, τ)     advance the states of X, and those of V by      Jacobian,
#                              the tangent equation along them                 subspace
#     shift!(P, X, s)          shift every state of X by the shifts s          relative orbits
#     dshift!(P, G, X, j)      the generator of shift j at every state of X    relative orbits
#     forced!(P, X, V, τ)      as tangent!, forced by ∂f/∂p (sensitivity.jl)   sensitivity
#
# shift! and dshift! are needed for relative periodic orbits only, and forced! for the sensitivity
# to a parameter only. For relative periodic orbits the system must be equivariant: the shifts
# commute with the flow, and f(S(s) x) = S(s) f(x).

"""
    propagate!(P, X, τ) -> X

Advance every state of the batch `X` in place by the time `τ`.
"""
function propagate! end

"""
    rhs!(P, F, X) -> F

The time derivative of the system at every state of the batch `X`, written into the batch `F`.
"""
function rhs! end

"""
    tangent!(P, X, V, τ) -> V

Advance every state of the batch `X` by the time `τ`, and the state of the batch `V` with the
same index by the tangent equation along it, both in place.
"""
function tangent! end

"""
    shift!(P, X, s) -> X

Shift every state of the batch `X` in place by the shifts `s`, one per continuous symmetry.
Relative periodic orbits only.
"""
function shift! end

"""
    dshift!(P, G, X, j) -> G

The generator of the `j`-th shift, the derivative of the shifted state with respect to `sⱼ` at
`sⱼ = 0`, at every state of the batch `X`, written into the batch `G`. Relative periodic orbits
only.
"""
function dshift! end
