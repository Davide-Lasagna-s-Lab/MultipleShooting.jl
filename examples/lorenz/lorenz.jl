# The orbit AB of the Lorenz system at σ = 10, ρ = 28, β = 8/3, the shortest one, from four seeds of
# a near recurrence: the search with the seeds in the two layouts, batched and sliced, the
# derivative of the orbit in ρ, and the subspace iteration along the loop.
#
#     julia --project=examples examples/lorenz/lorenz.jl
#
# The steps are those of the RK4 of Flows.jl, of equal length, at most dt. The tangent goes through
# the stages of the scheme linearised along those of the states, so that it is the derivative of
# the scheme. The system is small, so that the flows are made at every call.
include(joinpath(@__DIR__, "..", "common.jl"))

using MultipleShooting
using NKRoots: nkroot!, Trace
using LinearAlgebra
using Random
import Flows


# ---------------------------------------------------------------------------- #
# the equation, on one state                                                   #
# ---------------------------------------------------------------------------- #

const σ = 10.0
const β = 8/3

# the right-hand side f(x) at ρ, into dx
function lorenz!(dx, x, ρ)
    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
    return dx
end

# its linearisation at x applied to v, into dv
function linearised!(dv, x, v, ρ)
    dv[1] = σ*(v[2] - v[1])
    dv[2] = (ρ - x[3])*v[1] - v[2] - x[1]*v[3]
    dv[3] = x[2]*v[1] + x[1]*v[2] - β*v[3]
    return dv
end

# The states of x, one state or the columns of a matrix, advanced by τ: alone, by f, or with the
# vectors v, by the term vec!(t, x, dx, v, dv) of the vectors along the stages of the states
_steps(τ, dt) = Flows.TimeStepConstant(τ/max(1, ceil(Int, τ/dt)))

function advance!(f!, x, τ, dt)
    ϕ = Flows.flow(f!, Flows.RK4(x), _steps(τ, dt))
    return ϕ(x, (0, τ))
end

function advance!(f!, vec!, x, v, τ, dt)
    z = Flows.couple(x, v)
    ϕ = Flows.flow(Flows.couple(f!, vec!), Flows.RK4(z), _steps(τ, dt))
    ϕ(z, (0, τ))
    return v
end

# f, its linearisation, the same forced by ∂f/∂ρ = (0, x₁, 0), and the same forced by f/τ, on one
# state or column by column, as Flows.jl calls them
rhs(ρ)          = (t, X, dX) -> (foreach((dx, x) -> lorenz!(dx, x, ρ), eachcol(dX), eachcol(X)); dX)
tangent(ρ)      = (t, X, dX, V, dV) ->
                      (foreach((dv, x, v) -> linearised!(dv, x, v, ρ), eachcol(dV), eachcol(X), eachcol(V)); dV)
forced(ρ)       = (t, X, dX, V, dV) -> (tangent(ρ)(t, X, dX, V, dV); view(dV, 2, :) .+= view(X, 1, :); dV)
stretched(ρ, τ) = (t, X, dX, V, dV) -> (tangent(ρ)(t, X, dX, V, dV); dV .+= dX ./ τ; dV)


# ---------------------------------------------------------------------------- #
# the objects of the dynamics, on the two layouts                              #
# ---------------------------------------------------------------------------- #

# The objects of the dynamics at ρ, one type with the role R: the flow :G, the tangent :L, the
# right-hand side :F, the tangent forced by ∂f/∂ρ :Lp, and the derivative of the flow in τ :dG
struct Lorenz{R}
     ρ::Float64 # the parameter
    dt::Float64 # the largest time step
end

lorenz(ρ; dt=1e-3) = (G  = Lorenz{:G}(ρ, dt),
                      L  = Lorenz{:L}(ρ, dt),
                      F  = Lorenz{:F}(ρ, dt),
                      Lp = Lorenz{:Lp}(ρ, dt),
                      dG = Lorenz{:dG}(ρ, dt))

# ---- batched, the columns of a 3 × m matrix: all the states at once ----
(G::Lorenz{:G})(X::Batched, τ::Real)               = advance!(rhs(G.ρ), X, τ, G.dt)
(L::Lorenz{:L})(X::Batched, V::Batched, τ::Real)   = advance!(rhs(L.ρ), tangent(L.ρ), X, V, τ, L.dt)
(P::Lorenz{:Lp})(X::Batched, V::Batched, τ::Real)  = advance!(rhs(P.ρ), forced(P.ρ), X, V, τ, P.dt)
(F::Lorenz{:F})(dX::Batched, X::Batched)           = rhs(F.ρ)(0.0, X, dX)

# the derivative of the end points in τ: over s = t/τ the state solves dx/ds = τ f(x), and its
# derivative in τ the tangent equation forced by f/τ, from zero
(D::Lorenz{:dG})(X::Batched, V::Batched, τ::Real) =
    (V .= 0; advance!(rhs(D.ρ), stretched(D.ρ, τ), X, V, τ, D.dt))

# k′ directions, each along a copy of the states, the last along the states themselves
function (L::Lorenz{:L})(X::Batched, V::AbstractVector{<:Batched}, τ::Real)
    for a in eachindex(V)
        L(a == lastindex(V) ? X : copy(X), V[a], τ)
    end
    return V
end

# ---- sliced, a vector of 3-vectors: one state at a time ----
(G::Lorenz{:G})(X::Sliced, τ::Real)             = (foreach(x -> G(x, τ), X); X)
(L::Lorenz{:L})(X::Sliced, V::Sliced, τ::Real)  = (foreach((x, v) -> L(x, v, τ), X, V); V)
(F::Lorenz{:F})(dX::Sliced, X::Sliced)          = (foreach((f, x) -> F(f, x), dX, X); dX)
(D::Lorenz{:dG})(X::Sliced, V::Sliced, τ::Real) = (foreach((x, v) -> D(x, v, τ), X, V); V)


# ---------------------------------------------------------------------------- #
# the searches                                                                 #
# ---------------------------------------------------------------------------- #

# ---- the guess: four seeds along the closest return of a trajectory with a period between 1.4
#      and 1.7, where only AB lives, a distance of 0.42 at T = 1.57 ----
guess = ShootingVector([-4.3916429753   4.5595009656   3.6883670592  -3.8671374687;
                         2.1325664921   7.9944849552  -1.9178392035  -6.9073223838;
                        30.6127436095  12.6485708665  29.3130502493  11.6370396562], [1.57])

D  = lorenz(28.0)
kw = (krylov_dim=20, krylov_tol=1e-10, Δ=1.0, tol=1e-12, maxiter=50, verbose=false)

# ---- batched, with the column of the period of the scheme, dG ----
z       = copy(guess)
batched = Trace()
nkroot!(Residual(D.G), Jacobian(D.L, D.F, z; D.dG), z; kw..., callback=batched)

# ---- sliced, the same seeds as a vector of states ----
x      = ShootingVector(Sliced(collect.(eachcol(guess.X))), copy(guess.d))
sliced = Trace()
nkroot!(Residual(D.G), Jacobian(D.L, D.F, x; D.dG), x; kw..., callback=sliced)

@printf("batched:              T = %.10f in %d iterations\n", period(z), batched.iter[end])
@printf("sliced:               T = %.10f in %d iterations\n", period(x), sliced.iter[end])

# ---- the derivative of the orbit in ρ, and that of its period against central differences ----
dz = tangent_sensitivity(D.L, D.Lp, D.F, z; D.dG)

function T(ρ)
    P = lorenz(ρ)
    w = copy(z)
    nkroot!(Residual(P.G), Jacobian(P.L, P.F, w; P.dG), w; kw...)
    return period(w)
end

@printf("dT/dρ:                %.8f, central differences %.8f\n",
        period(dz), (T(28 + 1e-4) - T(28 - 1e-4))/2e-4)

# ---- the subspace iteration on the orbit, step by step, with one direction at every seed, the
#      unstable one, and with two, the unstable and the neutral ones: the turn of the bases ----
Random.seed!(1)
turns = map((1, 2)) do k′
    S = Subspace(D.L, z; k′=k′)
    [(iterate!(S); maxturn(S)) for _ in 1:60]
end


# ---------------------------------------------------------------------------- #
# the figure                                                                   #
# ---------------------------------------------------------------------------- #

# the m segments of z in the plane (x, z), nf points each
function segments(z; nf=60)
    X = copy(z.X)
    τ = period(z)/nsegments(z)
    p = [Matrix{Float64}(undef, 3, nf + 1) for _ in 1:nsegments(z)]
    for n in 0:nf
        foreach(i -> (p[i][:, n + 1] .= X[:, i]), 1:nsegments(z))
        D.G(X, τ/nf)
    end
    return p
end

fig = Figure(size=(1150, 330))

for (j, (title, v)) in enumerate((("the guess", guess), ("the orbit AB", z)))
    ax = Axis(fig[1, j]; title=@sprintf("%s, T = %.4f", title, period(v)), xlabel="x",
              ylabel=(j == 1 ? "z" : ""))
    for (i, p) in enumerate(segments(v))
        lines!(ax, p[1, :], p[3, :]; color=SERIES[i])
    end
    scatter!(ax, v.X[1, :], v.X[3, :]; color=INK, markersize=9)
end

ax = Axis(fig[1, 3]; title="the residual", xlabel="Newton iteration", yscale=log10)
scatterlines!(ax, batched.iter, batched.res; color=BLUE, label="batched")
scatterlines!(ax, sliced.iter, sliced.res; color=ORANGE, marker=:xcross, label="sliced")
axislegend(ax; position=:lb)

ax = Axis(fig[1, 4]; title="the turn of the bases", xlabel="subspace step", yscale=log10)
scatterlines!(ax, turns[1]; color=BLUE, label="k′ = 1")
scatterlines!(ax, turns[2]; color=ORANGE, label="k′ = 2")
axislegend(ax; position=(0.5, 0.55))

finish(fig, joinpath(@__DIR__, "lorenz.png"))
