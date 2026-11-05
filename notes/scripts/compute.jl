# The data of the figures of the note: the Lorenz orbit AB and the shortest relative periodic orbit
# of Kuramoto–Sivashinsky at L = 22, each from its guess in the tests, the Newton histories of the
# two searches, and the subspace iteration on the two orbits with their Floquet multipliers. Run
# from this folder,
#
#     julia --project=. compute.jl
#     julia --project=. plot.jl
#
# The first prints the numbers quoted in the note and writes data.jls, the second the figures into
# ../figures. They are two scripts since CairoMakie loads an FFT of its own, whose plan_rfft is
# ambiguous with that of FFTW, on which the propagator of KS relies.
using MultipleShootingV2
using NKRoots: nkroot!, Trace
using LinearAlgebra
using Random
using Printf
using Serialization

const TEST = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST, "lorenz.jl"))
include(joinpath(TEST, "ks.jl"))
include(joinpath(TEST, "data", "ks22.jl"))



# ---------------------------------------------------------------------------- #
# the two searches                                                             #
# ---------------------------------------------------------------------------- #

# ---- Lorenz, from the guess of the tests ----
Lz     = LorenzPropagator(1e-3, 28.0)
lguess = MultiVector([-4.3916429753   4.5595009656   3.6883670592  -3.8671374687;
                       2.1325664921   7.9944849552  -1.9178392035  -6.9073223838;
                      30.6127436095  12.6485708665  29.3130502493  11.6370396562], [1.57])
ab     = copy(lguess)
ltrace = Trace()
nkroot!(Residual(Lz), Jacobian(Lz, ab), ab; krylov_dim=20, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
        maxiter=50, verbose=false, callback=ltrace)

# ---- KS, from the guess of the tests ----
K      = KSPropagator()
ks     = copy(ks22_guess)
ktrace = Trace()
nkroot!(Residual(K), Jacobian(K, ks), ks; krylov_dim=60, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
        maxiter=40, verbose=false, callback=ktrace)

@printf("Lorenz  T = %.10f, %d iterations, %d products\n", period(ab), ltrace.iter[end], ltrace.products[end])
@printf("KS      T = %.8f, s = %.8f, %d iterations, %d products\n", period(ks), shift(ks, 1),
        ktrace.iter[end], ktrace.products[end])


# ---- the sensitivity of the period of AB to ρ ----
dz = tangent_sensitivity(Lz, ab)
@printf("Lorenz  dT/dρ = %.8f\n", period(dz))


# ---------------------------------------------------------------------------- #
# Floquet multipliers from the dense monodromy of the discrete loop            #
# ---------------------------------------------------------------------------- #

# M = S(s) ψ_m ⋯ ψ₁, each ψᵢ by the tangent of the n unit vectors along n copies of seed i
function monodromy(P, z)
    n = size(z.X, 1)
    m = nsegments(z)
    τ = period(z)/m
    M = Matrix{Float64}(I, n, n)

    for i in 1:m
        X = repeat(z.X[:, i:i], 1, n)
        ψ = tangent!(P, X, Matrix{Float64}(I, n, n), τ)
        M = ψ*M
    end

    # ---- the shift of the last end, as a matrix on its columns ----
    nshifts(z) > 0 && (M = shift!(P, M, [shift(z, 1)]))

    return M
end

function multipliers(P, z)
    Λ = eigvals(monodromy(P, z))
    return Λ[sortperm(abs.(Λ); rev=true)]
end

ΛL = multipliers(Lz, ab)
ΛK = multipliers(K, ks)
λL = log.(abs.(ΛL)) ./ period(ab)
λK = log.(abs.(ΛK)) ./ period(ks)

println("Lorenz multipliers ", ΛL)
println("Lorenz exponents   ", λL)
println("KS multipliers     ", ΛK[1:8])
println("KS exponents       ", λK[1:8])


# ---------------------------------------------------------------------------- #
# the subspace iteration                                                       #
# ---------------------------------------------------------------------------- #

# ε_p after each of r steps, and the distance of the columns of W from the span of the bases,
# the largest over the seeds
function history(P, z, k, r; p=1:k, W=())
    Random.seed!(2)
    S = Subspace(Jacobian(P, z); k=k)
    ε = zeros(r, length(p))
    d = zeros(r, length(W))
    m = nsegments(z)

    for j in 1:r
        iterate!(S)
        ε[j, :] .= [change(S; p=q) for q in p]

        # ---- the part of w outside the span at every seed, relative to w ----
        for (a, w) in enumerate(W)
            Vᵢ(i)  = reduce(hcat, [Vᵇ[:, i] for Vᵇ in S.V])
            d[j, a] = maximum(norm(w[:, i] - Vᵢ(i)*(Vᵢ(i)'*w[:, i]))/norm(w[:, i]) for i in 1:m)
        end
    end

    return ε, d
end

εL, _ = history(Lz, ab, 2, 8)
εK3, _ = history(K, ks, 3, 40; p=3:3)
εK4, dK = history(K, ks, 4, 40; p=3:3, W=(rhs!(K, similar(ks.X), ks.X),
                                          dshift!(K, similar(ks.X), ks.X, 1)))
εK6, _ = history(K, ks, 6, 40; p=3:3)

println("Lorenz ε₁, ε₂ ", εL)
println("KS ε₃ k = 3 ", εK3[[4, 8, 16, 24, 32, 40]])
println("KS ε₃ k = 4 ", εK4[[4, 8, 16, 24, 32, 40]])
println("KS ε₃ k = 6 ", εK6[[4, 8, 16, 24, 32, 40]])
println("KS f, g out ", dK[[4, 8, 16, 24, 32, 40], :])



# ---------------------------------------------------------------------------- #
# the trajectories of the guesses and the orbits, segment by segment           #
# ---------------------------------------------------------------------------- #

# the trajectory from each seed over its segment, sampled every dt
function segments(P, z; dt)
    τ = period(z)/nsegments(z)
    n = max(2, ceil(Int, τ/dt))

    return map(1:nsegments(z)) do i
        x = z.X[:, i:i]
        Y = zeros(size(x, 1), n + 1)
        Y[:, 1] .= vec(x)
        for j in 1:n
            propagate!(P, x, τ/n)
            Y[:, j + 1] .= vec(x)
        end
        Y
    end
end

# ---- everything the plots need ----
data = (lguess = (X = lguess.X, d = lguess.d, segments = segments(Lz, lguess; dt=0.002)),
        ab     = (X = ab.X, d = ab.d, segments = segments(Lz, ab; dt=0.002)),
        kguess = (X = ks22_guess.X, d = ks22_guess.d, segments = segments(K, ks22_guess; dt=0.05)),
        ks     = (X = ks.X, d = ks.d, segments = segments(K, ks; dt=0.05)),
        ltrace = (iter = ltrace.iter, res = ltrace.res, step = ltrace.step),
        ktrace = (iter = ktrace.iter, res = ktrace.res, step = ktrace.step),
        λL = λL, λK = λK, ΛL = ΛL, ΛK = ΛK,
        εL = εL, εK3 = εK3, εK4 = εK4, εK6 = εK6, dK = dK)

serialize(joinpath(@__DIR__, "data.jls"), data)
