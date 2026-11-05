# The figures of the note, from data.jls written by compute.jl (see there). Run from this folder,
#
#     julia --project=. plot.jl
using CairoMakie
using Printf
using Serialization

const OUT = joinpath(@__DIR__, "..", "figures")
mkpath(OUT)

set_theme!(fontsize=10,
           linewidth=1.2,
           Axis=(xgridvisible=false, ygridvisible=false))

const C = Makie.wong_colors()

D = deserialize(joinpath(@__DIR__, "data.jls"))

# the period of the scalars d, and the number of segments of the seeds X
T(z) = z.d[1]
m(z) = size(z.X, 2)

lguess = D.lguess
ab     = D.ab
kguess = D.kguess
ks     = D.ks
ltrace = D.ltrace
ktrace = D.ktrace
λL     = D.λL
λK     = D.λK
εL     = D.εL
εK3    = D.εK3
εK4    = D.εK4
εK6    = D.εK6
dK     = D.dK

# ---------------------------------------------------------------------------- #
# figure: the Lorenz guess and orbit                                           #
# ---------------------------------------------------------------------------- #

let fig = Figure(size=(430, 210))
    for (c, z, title) in ((1, lguess, "guess, T = 1.57"), (2, ab, @sprintf("orbit AB, T = %.4f", T(ab))))
        ax = Axis(fig[1, c]; xlabel="x", ylabel="z", title=title, titlefont=:regular,
                  aspect=DataAspect())
        for Y in z.segments
            lines!(ax, Y[1, :], Y[3, :]; color=C[1])
            scatter!(ax, [Y[1, end]], [Y[3, end]]; color=:white, strokecolor=C[1], strokewidth=1,
                     markersize=6)
        end
        scatter!(ax, z.X[1, :], z.X[3, :]; color=:black, markersize=6)
        c == 2 && hideydecorations!(ax; ticks=false, ticklabels=false)
        limits!(ax, -20, 20, 0, 50)
    end
    save(joinpath(OUT, "lorenz.pdf"), fig)
end


# ---------------------------------------------------------------------------- #
# figure: the KS guess and orbit in space and time                             #
# ---------------------------------------------------------------------------- #

let fig = Figure(size=(430, 300))
    x = (0:63) .* (22/64)
    for (c, z, title) in ((1, kguess, "guess, T = 16.8"),
                          (2, ks, @sprintf("orbit, T = %.2f", T(ks))))
        τ  = T(z)/m(z)
        Ys = z.segments
        U  = reduce(hcat, [Y[:, 1:end-1] for Y in Ys])
        t  = range(0, T(z); length=size(U, 2) + 1)[1:end-1]
        ax = Axis(fig[1, c]; xlabel="x", ylabel="t", title=title, titlefont=:regular)
        hm = heatmap!(ax, x, t, U; colormap=:balance, colorrange=(-3, 3), rasterize=4)
        hlines!(ax, τ .* (0:m(z) - 1); color=:black, linewidth=0.8, linestyle=:dash)
        c == 2 && hideydecorations!(ax; ticks=false, ticklabels=false)
        c == 2 && Colorbar(fig[1, 3], hm; label="u", width=8)
    end
    save(joinpath(OUT, "ks.pdf"), fig)
end


# ---------------------------------------------------------------------------- #
# figure: Newton histories                                                     #
# ---------------------------------------------------------------------------- #

let fig = Figure(size=(430, 190))
    ax = Axis(fig[1, 1]; xlabel="Newton iteration", ylabel="‖F(z)‖", yscale=log10)
    scatterlines!(ax, ltrace.iter, ltrace.res; color=C[1], markersize=6, label="Lorenz, m = 4")
    scatterlines!(ax, ktrace.iter, ktrace.res; color=C[2], marker=:rect, markersize=6,
                  label="KS, m = 4")
    hook = findall(==("hook"), ktrace.step)
    scatter!(ax, ktrace.iter[hook], ktrace.res[hook]; color=:white, strokecolor=C[2],
             strokewidth=1.2, marker=:rect, markersize=6)
    hook = findall(==("hook"), ltrace.step)
    scatter!(ax, ltrace.iter[hook], ltrace.res[hook]; color=:white, strokecolor=C[1],
             strokewidth=1.2, markersize=6)
    axislegend(ax; position=:lb, framevisible=false)
    save(joinpath(OUT, "newton.pdf"), fig)
end


# ---------------------------------------------------------------------------- #
# figure: the subspace iteration against the predicted rates                   #
# ---------------------------------------------------------------------------- #

# a line of slope -(λ_p - λ_{p+1}) τ per step in log, through the history at step r₀: the rate of
# the first p directions, an iteration of dimension p of their own
rate(λ, p, τ, ε, r₀, r) = ε[r₀] .* exp.(-(λ[p] - λ[p + 1]) .* τ .* (r .- r₀))

let fig = Figure(size=(430, 430))
    # ---- Lorenz, k = 2 ----
    τ  = T(ab)/m(ab)
    r  = 1:size(εL, 1)
    ax = Axis(fig[1, 1]; xlabel="subspace step r", ylabel="εₚ", yscale=log10,
              title="Lorenz AB, k = 2", titlefont=:regular, xticks=1:8)
    for (p, steps) in ((1, 1:8), (2, 1:3))
        scatterlines!(ax, r, replace(εL[:, p], 0.0 => NaN); color=C[p], markersize=5, label="p = $p")
        lines!(ax, steps, rate(λL, p, τ, εL[:, p], 1, steps); color=C[p], linestyle=:dash)
    end
    axislegend(ax; position=:lb, framevisible=false)
    ylims!(ax, 1e-16, 1e2)

    # ---- KS, ε₃ for k = 4: those of k = 3 and 6 are the same ----
    τ  = T(ks)/m(ks)
    r  = 1:size(εK4, 1)
    ax = Axis(fig[1, 2]; xlabel="subspace step r", ylabel="ε₃", yscale=log10,
              title="KS orbit, p = 3", titlefont=:regular)
    scatterlines!(ax, r, εK4[:, 1]; color=C[1], markersize=4, label="k = 4")
    lines!(ax, 4:32, rate(λK, 3, τ, εK4[:, 1], 8, 4:32); color=C[1], linestyle=:dash)
    axislegend(ax; position=:rt, framevisible=false)
    ylims!(ax, 1e-10, 1e1)

    # ---- KS, f and g against the span, k = 4 ----
    ax = Axis(fig[2, 1:2]; xlabel="subspace step r", ylabel="relative distance from the span",
              yscale=log10, title="KS orbit, k = 4", titlefont=:regular)
    scatterlines!(ax, r, dK[:, 1]; color=C[1], markersize=4, label="f, time derivative")
    scatterlines!(ax, r, dK[:, 2]; color=C[2], marker=:rect, markersize=4,
                  label="g, generator of the shift")
    axislegend(ax; position=:rt, framevisible=false)
    ylims!(ax, 1e-16, 1e0)

    save(joinpath(OUT, "subspace.pdf"), fig)
end
