using MultipleShootingV2
using NKRoots: nkroot!, gram
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "lorenz.jl"))
include(joinpath(@__DIR__, "ks.jl"))
include(joinpath(@__DIR__, "data", "ks22.jl"))

# ---- Lorenz, and a guess of its orbit AB: four seeds along the closest return of a trajectory
#      with a period between 1.4 and 1.7, where only AB lives, found once offline from (1, 1, 1)
#      after a transient of 20, sampled every 0.01: a distance of 0.42 at T = 1.57 ----
L     = LorenzPropagator(1e-3, 28.0)
F     = Residual(L)
guess = MultiVector([-4.3916429753   4.5595009656   3.6883670592  -3.8671374687;
                      2.1325664921   7.9944849552  -1.9178392035  -6.9073223838;
                     30.6127436095  12.6485708665  29.3130502493  11.6370396562], [1.57])

# the same vector in the sliced layout, a vector of states
sliced(z) = MultiVector(collect.(eachcol(z.X)), copy(z.d))

# the search from z, overwritten with the orbit
search!(z) = nkroot!(F, Jacobian(L, z), z; krylov_dim=20, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
                     maxiter=50, verbose=false)

# ---- Kuramoto–Sivashinsky at L = 22, and the search from the guess of its shortest relative
#      periodic orbit in data/ks22.jl ----
K = KSPropagator()

ks_search!(z) = nkroot!(Residual(K), Jacobian(K, z), z; krylov_dim=60, krylov_tol=1e-8, Δ=1.0,
                        tol=1e-10, maxiter=40, verbose=false)


@testset "multivector                              " begin
    Random.seed!(2)
    A = randn(4, 3)
    B = randn(4, 3)
    slice(X) = collect.(eachcol(X))
    flat(v)   = [vec(v.X isa Sliced ? reduce(hcat, v.X) : v.X); v.d]

    # ---- the same vectors in both layouts: the columns of a matrix, or a vector of states ----
    for (z, w) in ((MultiVector(copy(A), [1.5]), MultiVector(copy(B), [-0.5])),
                   (MultiVector(slice(A), [1.5]), MultiVector(slice(B), [-0.5])))
        @test nsegments(z) == 3
        @test period(z) == 1.5
        @test nshifts(z) == 0
        @test_throws BoundsError shift(z, 1)

        # ---- the inner product and the norm are those of the flattened vector ----
        @test dot(z, w) ≈ dot(flat(z), flat(w))
        @test norm(z) ≈ norm(flat(z))

        # ---- broadcasting in place, on the seeds and the scalars alike ----
        v = copy(w)
        v .= 2 .* z .+ v
        @test flat(v) ≈ 2 .* flat(z) .+ flat(w)
        v .= 2 .* z .- 0.5 .* w
        @test flat(v) ≈ 2 .* flat(z) .- 0.5 .* flat(w)
        v .= z
        @test flat(v) == flat(z)
        v .*= 3
        @test flat(v) ≈ 3 .* flat(z)

        # ---- and out of place ----
        u = z .+ w
        @test u isa MultiVector
        @test flat(u) ≈ flat(z) .+ flat(w)
        @test flat(abs2.(z .- w)) ≈ abs2.(flat(z) .- flat(w))
    end

    # ---- a relative periodic orbit: the period, then the shifts ----
    z = MultiVector(randn(4, 3), [1.5, 0.2, -0.7])
    @test period(z) == 1.5
    @test nshifts(z) == 2
    @test shift(z, 1) == 0.2
    @test shift(z, 2) == -0.7
    @test_throws BoundsError shift(z, 0)
    @test_throws BoundsError shift(z, 3)
end


@testset "search of AB, batched and sliced          " begin
    z  = copy(guess)
    zs = sliced(guess)
    @test search!(z).status === :converged
    @test search!(zs).status === :converged
    @test abs(period(z) - 1.5586522107) < 1e-6
    @test abs(period(zs) - period(z)) < 1e-9
end

# ---- the orbit AB, converged once for the tests below ----
ab = copy(guess)
search!(ab)


@testset "residual                                 " begin
    # ---- it vanishes on the orbit, in both layouts ----
    @test norm(F(similar(ab), ab)) < 1e-9
    @test norm(F(similar(sliced(ab)), sliced(ab))) < 1e-9

    # ---- off it, the two layouts agree ----
    Random.seed!(1)
    w = MultiVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    @test reduce(hcat, F(similar(sliced(w)), sliced(w)).X) ≈ F(similar(w), w).X
end


@testset "Jacobian                                 " begin
    # ---- off the orbit, the columns of the seeds against central differences of the residual ----
    Random.seed!(1)
    w = MultiVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    J = Jacobian(L, w)
    unit(k) = (e = zeros(13); e[k] = 1; MultiVector(reshape(e[1:12], 3, 4), e[13:13]))
    flat(v) = [vec(v.X); v.d]
    A = reduce(hcat, [flat(J*unit(k)) for k in 1:12])
    D = reduce(hcat, [flat(F(similar(w), w .+ 1e-6 .* unit(k)) .-
                           F(similar(w), w .- 1e-6 .* unit(k))) ./ 2e-6 for k in 1:12])
    @test norm(A[1:12, :] - D[1:12, :]) < 1e-7*norm(A[1:12, :])

    # ---- the phase row is f at the first seed ----
    @test A[13, :] ≈ [vec(lorenz_rhs!(zeros(3, 1), w.X[:, 1:1], L.ρ)); zeros(9)]

    # ---- the same product in the sliced layout ----
    v = MultiVector(randn(3, 4), [0.3])
    @test reduce(hcat, (Jacobian(L, sliced(w))*sliced(v)).X) ≈ (J*v).X
end


@testset "subspace iteration                       " begin
    # ---- on the orbit with k = 2, in both layouts from the same random bases ----
    Random.seed!(2)
    S  = Subspace(L, ab; k=2)
    Sₛ = Subspace(L, sliced(ab); k=2)
    for a in 1:2
        Sₛ.V[a] .= collect.(eachcol(S.V[a]))
    end
    for _ in 1:8
        iterate!(S)
        iterate!(Sₛ)
    end

    # ---- the bases settle, and contain the flow, the neutral direction ----
    @test change(S) < 1e-6
    f = lorenz_rhs!(similar(ab.X), ab.X, L.ρ)
    Vᵢ(i) = reduce(hcat, [Vᵃ[:, i] for Vᵃ in S.V])
    @test maximum(norm(f[:, i] - Vᵢ(i)*(Vᵢ(i)'*f[:, i]))/norm(f[:, i]) for i in 1:4) < 1e-6

    # ---- the two layouts agree ----
    @test all(reduce(hcat, Sₛ.V[a]) ≈ S.V[a] for a in 1:2)
end


@testset "tangent sensitivity                      " begin
    # ---- dT/dρ of AB against central differences of the period of AB at ρ = 28 ± 1e-4 ----
    dz = tangent_sensitivity(L, ab)
    T(ρ) = (P = LorenzPropagator(1e-3, ρ);
            z = copy(ab);
            nkroot!(Residual(P), Jacobian(P, z), z; krylov_dim=20, krylov_tol=1e-10, Δ=1.0,
                    tol=1e-12, maxiter=50, verbose=false);
            period(z))
    @test abs(period(dz) - (T(28 + 1e-4) - T(28 - 1e-4))/2e-4) < 1e-6*abs(period(dz))
end


@testset "search of the KS orbit at L = 22          " begin
    # ---- the orbit of Cvitanović, Davidchack & Siminos (2010), figure 6.1 (a), Tp = 16.3 and
    #      ℓp = 2.86, to the digits given there; the guess lies near its reflection -u(-x), which
    #      has the opposite shift ----
    z = copy(ks22_guess)
    @test ks_search!(z).status === :converged
    @test abs(period(z) - 16.3) < 0.05
    @test abs(shift(z, 1) + 2.86) < 0.005
end

# ---- the KS orbit, converged once for the test below ----
ks22 = copy(ks22_guess)
ks_search!(ks22)


@testset "subspace iteration on the KS orbit       " begin
    # ---- the first three directions settle, by about 0.58 a step, the rate of an iteration of
    #      dimension three, and the bases contain the generator of the shift, carried round the
    #      loop by the shift of the bases. f is left out: on the loop of the scheme it is neutral
    #      only up to the error of the time step ----
    Random.seed!(2)
    S = Subspace(K, ks22; k=4)
    for _ in 1:40
        iterate!(S)
    end
    @test change(S; p=3) < 1e-6
    g = dshift!(K, similar(ks22.X), ks22.X, 1)
    Vᵢ(i) = reduce(hcat, [Vᵃ[:, i] for Vᵃ in S.V])
    @test maximum(norm(g[:, i] - Vᵢ(i)*(Vᵢ(i)'*g[:, i]))/norm(g[:, i]) for i in 1:4) < 1e-10
end


@testset "subspace update to a tolerance           " begin
    # ---- steps until the first three directions are accurate, and no more than the cap ----
    Random.seed!(2)
    S = Subspace(K, ks22; k=4)
    MultipleShootingV2.update!(S, ks22; tol=1e-6, maxsteps=60, p=3)
    @test change(S; p=3) <= 1e-6 && S.steps < 60
    MultipleShootingV2.update!(S, ks22; tol=0.0, maxsteps=3)
    @test S.steps == 3
end


@testset "preconditioner                           " begin
    # ---- on the Lorenz orbit, the coarse matrix assembled from the subspace step is Wᵀ A W, column
    #      by column with the products of the Jacobian ----
    Random.seed!(1)
    J = Jacobian(L, ab)
    M = Preconditioner(L, ab; k=2)
    W = MultiVector[]
    for i in 1:4, a in 1:2
        w = MultiVector(zeros(3, 4), [0.0])
        w.X[:, i] .= M.S.U[a][:, i]
        push!(W, w)
    end
    push!(W, MultiVector(zeros(3, 4), [1.0]))
    E = reduce(hcat, [MultipleShootingV2._restrict(M, J*w) for w in W])
    @test norm(Matrix(M.E) - E) < 1e-12*norm(E)

    # ---- M⁻¹ against the dense -(I - W Wᵀ) + W E⁻¹ Wᵀ, and the root of the Gram matrix of the
    #      vectors M⁻¹ qⱼ against that matrix itself, for three orthonormal qⱼ ----
    flat(w)   = [vec(w.X); w.d]
    unflat(x) = MultiVector(reshape(x[1:12], 3, 4), x[13:13])
    Wd = reduce(hcat, flat.(W))
    x  = randn(13)
    Mx = -(x - Wd*(Wd'*x)) + Wd*(Matrix(M.E) \ (Wd'*x))
    @test norm(flat(ldiv!(similar(ab), M, unflat(x))) - Mx) < 1e-12*norm(Mx)

    Q = [unflat(q) for q in eachcol(Matrix(qr(randn(13, 3)).Q))]
    Y = reduce(hcat, [flat(ldiv!(similar(ab), M, q)) for q in Q])
    R = gram(M, Q)
    @test norm(R'*R - Y'*Y) < 1e-12*norm(Y'*Y)

    # ---- the KS search with the preconditioner, k = 4, reaches the orbit ----
    Random.seed!(2)
    z  = copy(ks22_guess)
    Jk = Jacobian(K, z)
    @test nkroot!(Residual(K), Jk, z; precond=Preconditioner(K, z; k=4), flexible=true, krylov_dim=60,
                  krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false).status === :converged
end
