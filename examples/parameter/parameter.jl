# The length ℓ of the domain as a parameter of the shortest relative periodic orbit of the
# Kuramoto–Sivashinsky equation at ℓ = 22: its derivative in ℓ, the branch followed to its fold
# and through it, and the search with ℓ free from a guess where no orbit is.
#
#     julia --project=examples examples/parameter/parameter.jl
#
# The equation and the objects of its dynamics are in ../ks/dynamics.jl. On x̃ = 2πx/ℓ the shift
# does not depend on ℓ, so that the forced tangent Lp gives the derivatives in ℓ in one
# integration of the seeds.
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "ks", "dynamics.jl"))

using NKRoots: nkroot!, Trace

# ---- the search at a fixed ℓ, from z, overwritten with the orbit ----
kw = (krylov_dim=60, krylov_tol=1e-10, Δ=1.0, tol=1e-10, maxiter=40, verbose=false)

function orbit!(z, ℓ; callback=info -> false)
    D = ks(ℓ)
    return nkroot!(Residual(D.G; D.S), Jacobian(D.L, D.F, z; D.S, D.dS), z; kw...,
                   callback=callback)
end

D = ks(22.0)
z = copy(GUESS)
orbit!(z, 22.0)


# ---------------------------------------------------------------------------- #
# the sensitivity                                                              #
# ---------------------------------------------------------------------------- #

# ---- dz/dℓ at ℓ = 22, by the forced tangent, against central differences of the orbits at
#      ℓ = 22 ± 1e-3 ----
dz = tangent_sensitivity(D.L, D.Lp, D.F, z; D.S, D.dS)
zp = copy(z)
zm = copy(z)
orbit!(zp, 22.0 + 1e-3)
orbit!(zm, 22.0 - 1e-3)

@printf("dT/dℓ:   %.6f, central differences %.6f\n",
        period(dz), (period(zp) - period(zm))/2e-3)
@printf("ds̃/dℓ:   %.6f, central differences %.6f\n",
        shift(dz, 1), (shift(zp, 1) - shift(zm, 1))/2e-3)


# ---------------------------------------------------------------------------- #
# the continuation                                                             #
# ---------------------------------------------------------------------------- #

# ---- to smaller domains, until the branch turns at a fold, the column of ℓ by Lp ----
periods = (0.5period(z), 2period(z))
y       = copy(z)
tofold  = continuation!(at, D.G, D.L, D.F, y, 22.0; D.Lp, D.S, D.dS, sense=-1, krylov_dim=60,
                        periods=periods)
ℓf      = tofold.p
@printf("to the fold:     %s at ℓ = %.4f in %d points, T = %.4f\n",
        tofold.status, ℓf, length(tofold.path.p), period(y))

# ---- through the fold, to ℓ = 22.5 on the other side of the branch ----
x        = copy(z)
through  = continuation!(at, D.G, D.L, D.F, x, 22.0; D.Lp, D.S, D.dS, sense=-1, target=22.5,
                         krylov_dim=60, periods=periods)
@printf("to the target:   %s at ℓ = %.4f in %d points, T = %.4f, s = %.4f\n",
        through.status, through.p, length(through.path.p), period(x), shift_x(x, through.p))


# ---------------------------------------------------------------------------- #
# the search with the parameter free                                           #
# ---------------------------------------------------------------------------- #

# ---- below the fold, at ℓ = ℓf - 0.05, there is no orbit near the one of the fold: the search
#      at that ℓ stalls, the search with ℓ free converges onto the branch ----
ℓg    = ℓf - 0.05
fixed = Trace()
out   = orbit!(copy(y), ℓg; callback=fixed)
@printf("fixed ℓ = %.4f:  %s in %d iterations, residual %.1e\n", ℓg, out.status, out.iter, out.res)

# the residual and ℓ after each number of iterations of the free search, from the same start,
# up to convergence
free = [freesearch!(at, D.G, D.L, D.F, copy(y), ℓg; D.Lp, D.S, D.dS, maxiter=0, krylov_dim=60)]
while free[end].status !== :converged && length(free) <= 10
    push!(free, freesearch!(at, D.G, D.L, D.F, copy(y), ℓg; D.Lp, D.S, D.dS,
                            maxiter=length(free), krylov_dim=60, krylov_tol=1e-8))
end
@printf("ℓ free:          %s in %d iterations, residual %.1e, at ℓ = %.4f\n",
        free[end].status, free[end].iter, free[end].res, free[end].p)


# ---------------------------------------------------------------------------- #
# the figure                                                                   #
# ---------------------------------------------------------------------------- #

fig = Figure(size=(1150, 360))

# ---- the branch, with the line of the sensitivity at ℓ = 22 ----
ax = Axis(fig[1, 1]; title="the branch through its fold", xlabel="ℓ", ylabel="T")
scatterlines!(ax, through.path.p, through.path.period; color=AQUA, label="continuation")
lines!(ax, [21.85, 22.1], period(z) .+ period(dz) .* [-0.15, 0.1]; color=ORANGE, linewidth=3,
       label="T + (dT/dℓ) δℓ")
scatter!(ax, [ℓf], [period(y)]; color=INK, marker=:diamond, markersize=12, label="the fold")
axislegend(ax; position=:rb)

# ---- the search below the fold, at fixed ℓ and with ℓ free ----
ax = Axis(fig[1, 2]; title=@sprintf("below the fold, ℓ = %.3f at the start", ℓg),
          xlabel="iteration", yscale=log10)
scatterlines!(ax, fixed.iter, fixed.res; color=BLUE, label="fixed ℓ")
scatterlines!(ax, 0:(length(free) - 1), [o.res for o in free]; color=ORANGE, label="ℓ free")
axislegend(ax; position=:lb)

# ---- the orbit at the target, on the other side of the branch ----
function spacetime(z, ℓ; nf=40)
    m = nsegments(z)
    G = Flow(ℓ, m)
    T = Transforms(m)
    X = copy(z.X)
    τ = period(z)/m
    u = zeros(nf, m, NGRID)
    for n in 1:nf
        u[n, :, :] .= grid!(T, X)'
        G(X, τ/nf)
    end
    return reshape(u, nf*m, NGRID)
end

u  = spacetime(x, through.p)
ax = Axis(fig[1, 3]; title=@sprintf("the orbit at ℓ = %.1f, T = %.2f", through.p, period(x)),
          xlabel="x", ylabel="t")
heatmap!(ax, range(0, through.p; length=NGRID), range(0, period(x); length=size(u, 1)), u';
         colormap=:balance, colorrange=(-3, 3))

finish(fig, joinpath(@__DIR__, "parameter.png"))
