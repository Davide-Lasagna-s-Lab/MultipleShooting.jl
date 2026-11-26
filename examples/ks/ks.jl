# The shortest relative periodic orbit of the Kuramoto–Sivashinsky equation at L = 22, that of
# Cvitanović, Davidchack & Siminos (2010), from four seeds of a near recurrence: a search, and the
# same search preconditioned.
#
#     julia --project=examples examples/ks/ks.jl
#
# The equation and the objects of its dynamics are in dynamics.jl.
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "dynamics.jl"))

using NKRoots: nkroot!, Trace
using Random


# ---------------------------------------------------------------------------- #
# the searches                                                                 #
# ---------------------------------------------------------------------------- #

D = ks(22.0)
R = Residual(D.G; D.S)

# ---- the search, with the tangent ----
z     = copy(GUESS)
plain = Trace()
out   = nkroot!(R, Jacobian(D.L, D.F, z; D.S, D.dS), z; krylov_dim=60, krylov_tol=1e-8, Δ=1.0,
                tol=1e-10, maxiter=40, verbose=false, callback=plain)
@printf("search:                 %s in %d iterations, %3d products, T = %.4f, s = %.4f\n",
        out.status, out.iter, plain.products[end], period(z), shift_x(z, 22))

# ---- the same, preconditioned on bases of four directions at every seed, which hold the unstable
#      one and the two neutral ones, of time and of the shift: five subspace steps from random
#      bases when it is built, then one at every Newton step ----
Random.seed!(1)
w   = copy(GUESS)
M   = Preconditioner(D.Lk, D.F, w; k′=4, D.S, D.dS, sweeps=5, tol=0.0, maxsteps=1, cap=:keep)
pre = Trace()
out = nkroot!(R, Jacobian(D.L, D.F, w; D.S, D.dS), w; precond=M, flexible=true, krylov_dim=60,
              krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false, callback=pre)
@printf("preconditioned search:  %s in %d iterations, %3d products, T = %.4f, s = %.4f\n",
        out.status, out.iter, pre.products[end], period(w), shift_x(w, 22))

# ---- the cost in tangent integrations of the orbit: one per product, and four per subspace
#      step, five when the preconditioner is built and one per update ----
cost = pre.products .+ 4 .* (5 .+ pre.preconditioner_updates)
@printf("tangent integrations:   %d without the preconditioner, %d with it\n",
        plain.products[end], cost[end])


# ---------------------------------------------------------------------------- #
# the figure                                                                   #
# ---------------------------------------------------------------------------- #

# u(x, t) over the m segments of z one after the other, nf frames each, on the grid, and τ
function spacetime(z; nf=40)
    m = nsegments(z)
    G = Flow(22.0, m)
    T = Transforms(m)
    X = copy(z.X)
    τ = period(z)/m
    u = zeros(nf, m, NGRID)
    for n in 1:nf
        u[n, :, :] .= grid!(T, X)'
        G(X, τ/nf)
    end

    return reshape(u, nf*m, NGRID), τ
end

fig = Figure(size=(900, 330))

for (j, (title, v)) in enumerate((("the guess", GUESS), ("the orbit", z)))
    u, τ = spacetime(v)
    t    = range(0, 4τ; length=size(u, 1))
    x    = range(0, 22; length=NGRID)
    ax   = Axis(fig[1, j]; title=@sprintf("%s, T = %.2f", title, period(v)), xlabel="x",
                ylabel=(j == 1 ? "t" : ""))
    heatmap!(ax, x, t, u'; colormap=:balance, colorrange=(-3, 3))
    hlines!(ax, τ .* (1:3); color=INK, linestyle=:dash, linewidth=1)
end

# ---- the residual against the cost, in tangent integrations of the orbit ----
ax = Axis(fig[1, 3]; title="the residual", xlabel="tangent integrations", yscale=log10)
scatterlines!(ax, plain.products, plain.res; color=BLUE, label="search")
scatterlines!(ax, cost, pre.res; color=ORANGE, label="preconditioned")
axislegend(ax; position=:lb)

finish(fig, joinpath(@__DIR__, "ks.png"))
