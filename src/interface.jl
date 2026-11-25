# The interface: what multiple shooting needs from the caller. Two things, both of types of the
# caller's choice, of which the package assumes nothing: the seeds, and the dynamics. This file
# declares the functions of the seeds, with no method, and the calls of the dynamics. The seeds
# have methods for two kinds of them, in seeds/: an array of numbers, a Batched (batched.jl), and
# a vector of states, a Sliced (sliced.jl). For seeds of another kind the caller writes them, and
# the objects of the dynamics are the caller's.
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
# The dynamics. Objects of the caller's types, each with one task, which it does when called, on
# all the states of an X at once, on threads or on a GPU, or one at a time. A search calls them as
#
#     G(X, τ)          the flow: every state of X advanced by the time τ, in place
#     L(X, V, τ)       the tangent: the states of X advanced, and those of V by the tangent
#                      equation along them, in place; V like X, or a vector of k′ of them
#     F(dX, X)         the right-hand side: f at every state of X, written into dX
#     S(X, s)          the shift: every state of X shifted by the shifts s, in place
#     dS(dX, X, j)     the generator of shift j at every state of X, written into dX
#     Lp(X, V, τ)      the forced tangent: as L, with the tangent equation forced by ∂f/∂p
#     dG(X, V, τ)      the derivative of the flow in τ: X advanced, and V the derivative in τ of
#                      the end points
#
# and uses no value they return. Each part of a search takes the objects it calls, and no other:
#
#     Residual(G; S)                       the flow, and the shift
#     Jacobian(L, F, z; S, dS, dG)         the tangent with one V like X, the right-hand side,
#                                          the shift and its generator, and the derivative of the
#                                          flow or, by default, F at the end points
#     JacobianFD(G, F, z; S, dS)           the flow, the right-hand side, the shift and its
#                                          generator
#     Subspace(L, z; k′, S)                the tangent with a vector of k′, and the shift
#     Preconditioner(L, F, z; k′, S, dS)   the tangent with a vector of k′, the right-hand side,
#                                          the shift and its generator
#     tangent_sensitivity(L, Lp, F, z; …)  those of the Jacobian, and the forced tangent
#
# so that an object may hold the workspaces of its task alone, made for it, and one object may
# also do several tasks. The shift and its generator are for relative periodic orbits only; the
# system must then be equivariant, the shifts commuting with the flow, and f(S(s) x) = S(s) f(x).
# The searches with a parameter, freesearch! and continuation!, move each object to the parameter
# p by a function family(x, p) of the caller's.


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
# internal: built on the interface of the seeds                                #
# ---------------------------------------------------------------------------- #

# the inner product of X and Y, the sum over their states
_dot(X, Y) = sum(dots(X, Y))
