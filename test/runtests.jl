using MultipleShooting
using NKRoots: nkroot!, gram
import NKRoots
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "lorenz.jl"))
include(joinpath(@__DIR__, "cycles.jl"))
include(joinpath(@__DIR__, "ks.jl"))
include(joinpath(@__DIR__, "data", "ks22.jl"))

# ---- Lorenz, and a guess of its orbit AB: four seeds along the closest return of a trajectory
#      with a period between 1.4 and 1.7, where only AB lives, found once offline from (1, 1, 1)
#      after a transient of 20, sampled every 0.01: a distance of 0.42 at T = 1.57 ----
Lz    = lorenz(28.0)
R     = Residual(Lz.G)
guess = ShootingVector([-4.3916429753   4.5595009656   3.6883670592  -3.8671374687;
                      2.1325664921   7.9944849552  -1.9178392035  -6.9073223838;
                     30.6127436095  12.6485708665  29.3130502493  11.6370396562], [1.57])

# the same vector with seeds that are not an array, a Sliced, a vector of states
sliced(z) = ShootingVector(Sliced(collect.(eachcol(z.X))), copy(z.d))

# the search from z, overwritten with the orbit
search!(z) = nkroot!(R, Jacobian(Lz.L, Lz.F, z; Lz.dG), z; krylov_dim=20, krylov_tol=1e-8, Δ=1.0,
                     tol=1e-10, maxiter=50, verbose=false)

# ---- Kuramoto–Sivashinsky at L = 22, and the search from the guess of its shortest relative
#      periodic orbit in data/ks22.jl ----
K = ks()

ks_search!(z) = nkroot!(Residual(K.G; K.S), Jacobian(K.L, K.F, z; K.S, K.dS), z; krylov_dim=60,
                        krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false)


@testset "shooting vector                          " begin
    Random.seed!(2)
    A = randn(4, 3)
    B = randn(4, 3)
    slice(X) = Sliced(collect.(eachcol(X)))
    flat(v)   = [vec(v.X isa Sliced ? reduce(hcat, v.X) : v.X); v.d]

    # ---- the same vectors with both batches: the columns of a matrix, or a vector of states ----
    for (z, w) in ((ShootingVector(copy(A), [1.5]), ShootingVector(copy(B), [-0.5])),
                   (ShootingVector(slice(A), [1.5]), ShootingVector(slice(B), [-0.5])))
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
        @test u isa ShootingVector
        @test flat(u) ≈ flat(z) .+ flat(w)
        @test flat(abs2.(z .- w)) ≈ abs2.(flat(z) .- flat(w))
    end

    # ---- a relative periodic orbit: the period, then the shifts ----
    z = ShootingVector(randn(4, 3), [1.5, 0.2, -0.7])
    @test period(z) == 1.5
    @test nshifts(z) == 2
    @test shift(z, 1) == 0.2
    @test shift(z, 2) == -0.7
    @test_throws BoundsError shift(z, 0)
    @test_throws BoundsError shift(z, 3)

    # ---- the states moved forward by one slot, and back, with both batches ----
    for X in (copy(A), slice(A))
        cols(Y) = Y isa Sliced ? reduce(hcat, Y) : Y
        @test cols(MultipleShooting.roll_forward!(X)) == A[:, [3, 1, 2]]
        @test cols(MultipleShooting.roll_backward!(X)) == A
        @test cols(MultipleShooting.roll_backward!(X)) == A[:, [2, 3, 1]]
    end
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
    # ---- it vanishes on the orbit, with both batches ----
    @test norm(R(similar(ab), ab)) < 1e-9
    @test norm(R(similar(sliced(ab)), sliced(ab))) < 1e-9

    # ---- off it, the two layouts agree ----
    Random.seed!(1)
    w = ShootingVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    @test reduce(hcat, R(similar(sliced(w)), sliced(w)).X) ≈ R(similar(w), w).X
end


@testset "Jacobian                                 " begin
    # ---- off the orbit, the columns of the seeds against central differences of the residual ----
    Random.seed!(1)
    w = ShootingVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    J = Jacobian(Lz.L, Lz.F, w; Lz.dG)
    unit(k) = (e = zeros(13); e[k] = 1; ShootingVector(reshape(e[1:12], 3, 4), e[13:13]))
    flat(v) = [vec(v.X); v.d]
    A = reduce(hcat, [flat(J*unit(k)) for k in 1:12])
    D = reduce(hcat, [flat(R(similar(w), w .+ 1e-6 .* unit(k)) .-
                           R(similar(w), w .- 1e-6 .* unit(k))) ./ 2e-6 for k in 1:12])
    @test norm(A[1:12, :] - D[1:12, :]) < 1e-7*norm(A[1:12, :])

    # ---- the phase row is f at the first seed ----
    @test A[13, :] ≈ [vec(lorenz_rhs!(zeros(3, 1), w.X[:, 1:1], Lz.F.ρ)); zeros(9)]

    # ---- the same product with the batch of states ----
    v = ShootingVector(randn(3, 4), [0.3])
    @test reduce(hcat, (Jacobian(Lz.L, Lz.F, sliced(w); Lz.dG)*sliced(v)).X) ≈ (J*v).X
end


@testset "period column                            " begin
    # ---- with a step of 10⁻², the column of the period against central differences of the residual
    #      in T at the same number of steps: the derivative of the flow dG of Lorenz, the forced
    #      tangent, is the derivative of the integrator, the default, f at the end points, only up
    #      to the error of the scheme ----
    Q = lorenz(28.0; dt=1e-2)
    Random.seed!(1)
    w  = ShootingVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    eT = ShootingVector(zeros(3, 4), [1.0])
    flat(v)     = [vec(v.X); v.d]
    residual(y) = flat(Residual(Q.G)(similar(y), y))
    D = (residual(w .+ 1e-6 .* eT) .- residual(w .- 1e-6 .* eT)) ./ 2e-6
    @test norm(flat(Jacobian(Q.L, Q.F, w; Q.dG)*eT) - D) < 1e-8*norm(D)
    @test norm(flat(Jacobian(Q.L, Q.F, w)*eT) - D) > 1e-6*norm(D)

    # ---- the same with the batch of states ----
    @test reduce(hcat, (Jacobian(Q.L, Q.F, sliced(w); Q.dG)*sliced(eT)).X) ≈ (Jacobian(Q.L, Q.F, w; Q.dG)*eT).X
    @test reduce(hcat, (Jacobian(Q.L, Q.F, sliced(w))*sliced(eT)).X) ≈ (Jacobian(Q.L, Q.F, w)*eT).X
end


@testset "subspace iteration                       " begin
    # ---- on the orbit with k′ = 2, with both batches from the same random bases ----
    Random.seed!(2)
    S  = Subspace(Lz.L, ab; k′=2)
    Sₛ = Subspace(Lz.L, sliced(ab); k′=2)
    for a in 1:2
        copyto!(Sₛ.V[a], Sliced(collect.(eachcol(S.V[a]))))
    end
    iterate!(S)
    iterate!(Sₛ)

    # ---- the turn of the first step: the sines of the principal angles between the spans before
    #      and after it, against a dense SVD of Zᵢ at every point. Two planes of a space of three
    #      dimensions share a line, so that the smallest angle vanishes, and the turn resolves it
    #      to rounding, while from random bases the other is large ----
    Uᵢ(i) = reduce(hcat, [Uᵃ[:, i] for Uᵃ in S.U])
    Vᵢ(i) = reduce(hcat, [Vᵃ[:, i] for Vᵃ in S.V])
    @test all(S.σ[:, i] ≈ sort(svdvals(Vᵢ(i) - Uᵢ(i)*(Uᵢ(i)'*Vᵢ(i)))) for i in 1:4)
    @test maxturn(S; k=1) < 1e-14
    @test maxturn(S; k=2) > 0.5

    # ---- the bases settle, their turn falls to rounding, and they contain the flow, the neutral
    #      direction ----
    for _ in 2:8
        iterate!(S)
        iterate!(Sₛ)
    end
    @test maxturn(S) < 1e-13
    f = lorenz_rhs!(similar(ab.X), ab.X, Lz.F.ρ)
    @test maximum(norm(f[:, i] - Vᵢ(i)*(Vᵢ(i)'*f[:, i]))/norm(f[:, i]) for i in 1:4) < 1e-6

    # ---- the two layouts agree ----
    @test all(reduce(hcat, Sₛ.V[a]) ≈ S.V[a] for a in 1:2)
    @test Sₛ.E ≈ S.E

    # ---- a step leaves the end points of the segments, bases that span the images that the
    #      tangent gives of each direction alone, at the point they reach, and the Galerkin blocks of the
    #      bases before it, Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ ----
    τ = period(ab)/4
    W = map(S.V) do Vᵃ
        Lz.L(copy(ab.X), copy(Vᵃ), τ)
    end
    Wᵢ(i) = reduce(hcat, [Wᵇ[:, i] for Wᵇ in W])
    iterate!(S)
    @test all(Vᵢ(i % 4 + 1)*(Vᵢ(i % 4 + 1)'*Wᵢ(i)) ≈ Wᵢ(i) for i in 1:4)
    @test S.tmp ≈ Lz.G(copy(ab.X), τ)
    @test all(S.E[:, :, i] ≈ Uᵢ(i % 4 + 1)'*reduce(hcat, [Wᵇ[:, i] for Wᵇ in W]) for i in 1:4)

    # ---- the tangent advances the batches of several directions along the states, each as on the
    #      batch alone, and leaves the end points in the states ----
    X = copy(ab.X)
    V = copy.(S.V)
    W = [Lz.L(copy(ab.X), copy(Vᵃ), τ) for Vᵃ in S.V]
    Lz.L(X, V, τ)
    @test X ≈ Lz.G(copy(ab.X), τ)
    @test all(V[a] ≈ W[a] for a in 1:2)

    # ---- on bases that have settled, the update passes the test after one step ----
    MultipleShooting.update!(S, ab; tol=1e-12)
    @test S.steps == 1
end


@testset "tangent sensitivity                      " begin
    # ---- dT/dρ of AB against central differences of the period of AB at ρ = 28 ± 1e-4 ----
    dz = tangent_sensitivity(Lz.L, Lz.Lp, Lz.F, ab; Lz.dG)
    T(ρ) = (P = lorenz(ρ);
            z = copy(ab);
            nkroot!(Residual(P.G), Jacobian(P.L, P.F, z; P.dG), z; krylov_dim=20, krylov_tol=1e-10,
                    Δ=1.0, tol=1e-12, maxiter=50, verbose=false);
            period(z))
    @test abs(period(dz) - (T(28 + 1e-4) - T(28 - 1e-4))/2e-4) < 1e-6*abs(period(dz))
end


@testset "GMRES and Ritz values                    " begin
    # ---- a matrix of size 30, not normal, with five distinct eigenvalues: GMRES ends after five
    #      iterations, the degree of its minimal polynomial, with the residual of the solution as
    #      its last, and its Ritz values are then the five eigenvalues ----
    Random.seed!(3)
    V = I + 0.1 .* randn(30, 30)
    A = V*Diagonal(repeat(1.0:5.0, 6))/V
    b = randn(30)
    g = gmres((out, v) -> mul!(out, A, v), b; tol=1e-10, maxiter=30)
    @test length(g.residuals) == 6
    @test abs(g.residuals[end] - norm(b - A*g.x)/norm(b)) < 1e-12
    @test ritz_values(g.H) ≈ 1:5
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
    # ---- from random bases of four directions the span turns far in the first step ----
    Random.seed!(2)
    S = Subspace(K.L, ks22; k′=4, K.S)
    iterate!(S)
    @test maxturn(S; k=1) > 0.5

    # ---- the span of the three leading directions settles within the four: the three smallest
    #      angles fall to rounding, while the fourth, that of the guard vector, still turns. The
    #      angles come in order, and the bases contain the generator of the shift, carried round
    #      the loop by the shift of the bases. f is left out: on the loop of the scheme it is
    #      neutral only up to the error of the time step ----
    for _ in 2:40
        iterate!(S)
    end
    @test maxturn(S; k=3) < 1e-13
    @test maxturn(S; k=4) > 100*maxturn(S; k=3)
    @test maxturn(S; k=1) <= maxturn(S; k=3) <= maxturn(S; k=4)
    @test_throws ArgumentError maxturn(S; k=5)
    g = K.dS(similar(ks22.X), ks22.X, 1)
    Vᵢ(i) = reduce(hcat, [Vᵃ[:, i] for Vᵃ in S.V])
    @test maximum(norm(g[:, i] - Vᵢ(i)*real.(Vᵢ(i)'*g[:, i]))/norm(g[:, i]) for i in 1:4) < 1e-10
end


@testset "subspace update to a tolerance           " begin
    # ---- steps until the turn of the first three directions passes the test, and no more than
    #      the cap ----
    Random.seed!(2)
    S = Subspace(K.L, ks22; k′=4, K.S)
    MultipleShooting.update!(S, ks22; tol=1e-6, k=3, maxsteps=60)
    @test maxturn(S; k=3) <= 1e-6 && S.steps < 60
    @test_throws ErrorException MultipleShooting.update!(S, ks22; tol=0.0, maxsteps=3)
    @test S.steps == 3

    # ---- with cap = :keep the cap is no error: tol = 0 makes exactly maxsteps steps ----
    MultipleShooting.update!(S, ks22; tol=0.0, maxsteps=3, cap=:keep)
    @test S.steps == 3
    @test_throws ArgumentError MultipleShooting.update!(S, ks22; tol=0.0, cap=:stop)
    @test_throws ArgumentError MultipleShooting.update!(S, ks22; tol=1e-6, k=5)
    @test_throws MethodError MultipleShooting.update!(S, ks22; tol=1e-6, p=3)
end


@testset "preconditioner                           " begin
    # ---- on the Lorenz orbit, the coarse matrix assembled from the subspace step is Wᵀ A W, column
    #      by column with the products of the Jacobian ----
    Random.seed!(1)
    J = Jacobian(Lz.L, Lz.F, ab; Lz.dG)
    M = Preconditioner(Lz.L, Lz.F, ab; k′=2)
    W = ShootingVector[]
    for i in 1:4, a in 1:2
        w = ShootingVector(zeros(3, 4), [0.0])
        w.X[:, i] .= M.S.U[a][:, i]
        push!(W, w)
    end
    push!(W, ShootingVector(zeros(3, 4), [1.0]))
    E = reduce(hcat, [MultipleShooting._restrict(M, J*w) for w in W])
    @test norm(Matrix(M.E)[:, 1:8] - E[:, 1:8]) < 1e-12*norm(E)

    # ---- the column of the period of E is f at the end points, that of the Jacobian the
    #      derivative of the integrator: they differ by the error of the scheme ----
    @test norm(Matrix(M.E)[:, 9] - E[:, 9]) < 1e-6*norm(E)

    # ---- M⁻¹ against the dense -(I - W Wᵀ) + W E⁻¹ Wᵀ, and the root of the Gram matrix of the
    #      vectors M⁻¹ qⱼ against that matrix itself, for three orthonormal qⱼ ----
    flat(w)   = [vec(w.X); w.d]
    unflat(x) = ShootingVector(reshape(x[1:12], 3, 4), x[13:13])
    Wd = reduce(hcat, flat.(W))
    x  = randn(13)
    Mx = -(x - Wd*(Wd'*x)) + Wd*(Matrix(M.E) \ (Wd'*x))
    @test norm(flat(ldiv!(similar(ab), M, unflat(x))) - Mx) < 1e-12*norm(Mx)

    Q = [unflat(q) for q in eachcol(Matrix(qr(randn(13, 3)).Q))]
    Y = reduce(hcat, [flat(ldiv!(similar(ab), M, q)) for q in Q])
    R = gram(M, Q)
    @test norm(R'*R - Y'*Y) < 1e-12*norm(Y'*Y)

    # ---- a solve with the coarse matrix whose relative residual is above the tolerance is an
    #      error: here the factors are those of half the matrix, and the residual is one ----
    B = Preconditioner(Lz.L, Lz.F, ab; k′=2)
    B.E.nzval .*= 2
    @test_throws ErrorException ldiv!(similar(ab), B, unflat(x))
    @test_throws ErrorException gram(B, Q)

    # ---- the KS search with the preconditioner reaches the orbit, with k′ = 4 and the test on
    #      the three leading directions: away from the orbit the fourth and the fifth exponents
    #      may pair, and the fourth direction, a guard vector here, never settles ----
    Random.seed!(2)
    z  = copy(ks22_guess)
    Jk = Jacobian(K.L, K.F, z; K.S, K.dS)
    Mk = Preconditioner(K.L, K.F, z; k′=4, k=3, K.S, K.dS)
    @test nkroot!(Residual(K.G; K.S), Jk, z; precond=Mk, flexible=true, krylov_dim=60, krylov_tol=1e-8,
                  Δ=1.0, tol=1e-10, maxiter=40, verbose=false).status === :converged
end


@testset "preconditioner: steps per update, reuse   " begin
    # ---- five sweeps and one step when built, then a fixed number of steps per update ----
    Random.seed!(2)
    z = copy(ks22_guess)
    M = Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS, sweeps=5, tol=0.0, maxsteps=1, cap=:keep)
    @test M.S.steps == 1
    NKRoots.update!(M, z)
    @test M.S.steps == 1 && M.updated == [true]

    # ---- at a fixed point, the steps to a tolerance on the leading directions stop once their
    #      turn passes the test ----
    N = Preconditioner(K.L, K.F, z; k′=4, k=3, K.S, K.dS, sweeps=5, tol=1e-6, maxsteps=40, cap=:keep)
    NKRoots.update!(N, z)
    @test maxturn(N.S; k=3) <= 1e-6 && N.S.steps < 40

    # ---- every = 2: updates at the Newton steps 1 and 3, and at step 2 the same E with no step ----
    U = Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS, sweeps=5, tol=0.0, maxsteps=1, cap=:keep, every=2)
    NKRoots.update!(U, z)
    E = copy(U.E)
    w = copy(z)
    w.X .+= 1e-3 .* randn(size(w.X))
    NKRoots.update!(U, w)
    @test U.E == E && U.S.z === z
    NKRoots.update!(U, w)
    @test U.updated == [true, false, true] && U.S.z === w && U.E != E

    # ---- every = 0: the first Newton step only ----
    Z = Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS, sweeps=5, tol=0.0, maxsteps=1, cap=:keep, every=0)
    for _ in 1:3
        NKRoots.update!(Z, w)
    end
    @test Z.updated == [true, false, false]
    @test_throws ArgumentError Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS, every=-1)
    @test_throws ArgumentError Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS, k=5)

    # ---- a relative periodic orbit with no shift, or no generator, is refused ----
    @test_throws ArgumentError Preconditioner(K.L, K.F, z; k′=4, K.S)
    @test_throws ArgumentError Subspace(K.L, z; k′=4)
    @test_throws ArgumentError Jacobian(K.L, K.F, z; K.dS)
    @test_throws ArgumentError Residual(K.G)(similar(z), z)

    # ---- the size of the bases has no default: a call with k alone fails, not read as k′ ----
    @test_throws UndefKeywordError Preconditioner(K.L, K.F, z; k=4, K.S, K.dS)
    @test_throws UndefKeywordError Subspace(K.L, z; k=4, K.S)

    # ---- the KS search with two steps per update and with the preconditioner of every second step
    #      both reach the orbit ----
    for (every, maxsteps) in ((1, 2), (2, 1))
        Random.seed!(2)
        u = copy(ks22_guess)
        P = Preconditioner(K.L, K.F, u; k′=4, K.S, K.dS, sweeps=5, tol=0.0, maxsteps=maxsteps, cap=:keep,
                           every=every)
        @test nkroot!(Residual(K.G; K.S), Jacobian(K.L, K.F, u; K.S, K.dS), u; precond=P, flexible=true,
                      krylov_dim=60, krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40,
                      verbose=false).status === :converged
    end
end


@testset "Jacobian by finite differences           " begin
    # ---- off the orbit, its products against those of the tangent equation, with both batches ----
    Random.seed!(1)
    w = ShootingVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    v = ShootingVector(randn(3, 4), [0.3])
    flat(u) = [vec(u.X); u.d]
    J  = JacobianFD(Lz.G, Lz.F, w)
    Jv = flat(Jacobian(Lz.L, Lz.F, w; Lz.dG)*v)
    @test norm(flat(J*v) - Jv) < 1e-6*norm(Jv)
    @test norm(flat(JacobianFD(Lz.G, Lz.F, w; difference=:central)*v) - Jv) < 1e-8*norm(Jv)
    @test reduce(hcat, (JacobianFD(Lz.G, Lz.F, sliced(w))*sliced(v)).X) ≈ (J*v).X

    # ---- the searches of AB and of the KS orbit, with a shift, find the same orbits ----
    z = copy(guess)
    @test nkroot!(R, JacobianFD(Lz.G, Lz.F, z), z; krylov_dim=20, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
                  maxiter=50, verbose=false).status === :converged
    @test abs(period(z) - period(ab)) < 1e-8
    z = copy(ks22_guess)
    @test nkroot!(Residual(K.G; K.S), JacobianFD(K.G, K.F, z; K.S, K.dS), z; krylov_dim=60,
                  krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false).status === :converged
    @test abs(period(z) - period(ks22)) < 1e-6
    @test abs(shift(z, 1) - shift(ks22, 1)) < 1e-6
end


@testset "bounds on the period                     " begin
    # ---- within the bounds the residual is the same, outside it is infinite, with no integration ----
    @test Residual(Lz.G; bounds=(1.0, 2.0))(similar(ab), ab).X == R(similar(ab), ab).X
    @test norm(Residual(Lz.G; bounds=(1.0, 1.5))(similar(ab), ab)) == Inf
    @test norm(Residual(Lz.G; bounds=(1.6, 2.0))(similar(ab), ab)) == Inf

    # ---- a period that is not positive is out of the default bounds ----
    for T in (0.0, -period(ab))
        z = ShootingVector(copy(ab.X), [T])
        @test norm(R(similar(z), z)) == Inf
    end
end


@testset "a parameter as an unknown, two cycles     " begin
    C = cycles(-0.3)
    outer(μ)  = (1 + sqrt(1 + 4μ))/2
    inner(μ)  = (1 - sqrt(1 + 4μ))/2
    radius²(z) = sum(abs2, z.X[:, 1])

    # ---- below the fold there is no orbit: from the circle of the fold at μ = -0.3 the search at
    #      a fixed parameter does not converge, and the search with the parameter free lands on a
    #      circle at a parameter above the fold ----
    z = circle(0.5)
    @test nkroot!(Residual(C.G), Jacobian(C.L, C.F, z), z; krylov_dim=9, krylov_tol=1e-10, Δ=0.1,
                  tol=1e-10, maxiter=50, verbose=false).status !== :converged
    z   = circle(0.5)
    out = freesearch!(at, C.G, C.L, C.F, z, -0.3; C.Lp, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :converged
    @test out.p > -0.25
    @test norm(Residual(cycles(out.p).G)(similar(z), z)) < 1e-10
    @test abs(radius²(z) - outer(out.p)) < 1e-6 || abs(radius²(z) - inner(out.p)) < 1e-6

    # ---- the column of the parameter from the forced tangent equation and from the central
    #      difference of the residual, off an orbit, and the same search with the second ----
    z = circle(0.6)
    Rμ(u, μ) = Residual(cycles(μ).G)(similar(u), u)
    c = MultipleShooting._forced(cycles(-0.2).Lp, nothing, z)
    @test norm(c .- (Rμ(z, -0.2 + 1e-5) .- Rμ(z, -0.2 - 1e-5)) ./ 2e-5) < 1e-8
    z = circle(0.5)
    @test abs(freesearch!(at, C.G, C.L, C.F, z, -0.3; krylov_tol=1e-10, krylov_dim=10).p - out.p) < 1e-8

    # ---- on an orbit it stays where it is, up to the error of the scheme ----
    z   = circle(outer(-0.1))
    out = freesearch!(at, C.G, C.L, C.F, z, -0.1; C.Lp, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :converged
    @test abs(out.p + 0.1) < 1e-8

    # ---- a trial parameter out of bounds is refused, and the search collapses ----
    z = circle(0.5)
    @test freesearch!(at, C.G, C.L, C.F, z, -0.3; C.Lp, bounds=(-0.35, -0.26), krylov_tol=1e-10,
                      krylov_dim=10).status === :collapsed

    # ---- with no target the continuation from the outer circle, the parameter shrinking, ends at
    #      the fold, μ = -1/4 and r² = 1/2 ----
    z   = circle(outer(-0.1))
    out = continuation!(at, C.G, C.L, C.F, z, -0.1; C.Lp, sense=-1, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :fold
    @test abs(out.p + 0.25) < 1e-6
    @test abs(radius²(z) - 0.5) < 2e-2
    @test all(<(0), out.path.slope[1:(end - 1)])

    # ---- with a target above it, it passes the fold and ends on the inner circle at the target ----
    z   = circle(outer(-0.1))
    out = continuation!(at, C.G, C.L, C.F, z, -0.1; C.Lp, sense=-1, target=-0.05, krylov_tol=1e-10,
                        krylov_dim=10)
    @test out.status === :target
    @test abs(out.p + 0.05) < 1e-9
    @test abs(radius²(z) - inner(-0.05)) < 1e-6
    @test minimum(out.path.p) < -0.24

    # ---- and with the parameter growing it ends on the outer circle at the same target ----
    z   = circle(outer(-0.1))
    out = continuation!(at, C.G, C.L, C.F, z, -0.1; C.Lp, target=-0.05, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :target
    @test abs(radius²(z) - outer(-0.05)) < 1e-6
    @test_throws ArgumentError continuation!(at, C.G, C.L, C.F, z, -0.05; C.Lp, sense=2)
end


@testset "continuation of AB and of the KS orbit    " begin
    # ---- AB from ρ = 28 to 28.5 against the search at 28.5, and the first slope dT/dρ against
    #      the tangent sensitivity ----
    z   = copy(ab)
    out = continuation!(at, Lz.G, Lz.L, Lz.F, z, 28.0; Lz.Lp, Lz.dG, target=28.5, krylov_tol=1e-10,
                        krylov_dim=20)
    w   = copy(ab)
    L₂  = lorenz(28.5)
    nkroot!(Residual(L₂.G), Jacobian(L₂.L, L₂.F, w; L₂.dG), w; krylov_dim=20, krylov_tol=1e-10, Δ=1.0,
            tol=1e-12, maxiter=50, verbose=false)
    @test out.status === :target
    @test abs(period(z) - period(w)) < 1e-8
    @test abs((out.path.period[2] - out.path.period[1])/(out.path.p[2] - out.path.p[1]) -
              period(tangent_sensitivity(Lz.L, Lz.Lp, Lz.F, ab; Lz.dG))) <
          1e-2*abs(period(tangent_sensitivity(Lz.L, Lz.Lp, Lz.F, ab; Lz.dG)))

    # ---- the Newton homotopy from the guess of AB, the amplitude of its residual from 1 to 0 ----
    z   = copy(guess)
    R₀  = R(similar(z), z)
    out = continuation!((x, a) -> x, Lz.G, Lz.L, Lz.F, z, 1.0; Lz.dG, sense=-1, target=0.0,
                        krylov_tol=1e-10, krylov_dim=20, residual=(r, u, a) -> (R(r, u); r .= r .- a .* R₀))
    @test out.status === :target
    @test norm(R(similar(z), z)) < 1e-8
    @test abs(period(z) - period(ab)) < 1e-6

    # ---- the KS orbit in the logarithm of the length of the domain, with the preconditioner and
    #      bounds on the period, from L = 22 to 22.1 against the search there. The length is in
    #      the shift too, so that the column is the central difference ----
    Random.seed!(2)
    z   = copy(ks22)
    out = continuation!((x, λ) -> at(x, exp(λ)), K.G, K.L, K.F, z, log(22.0); K.S, K.dS,
                        target=log(22.1), precond=Preconditioner(K.L, K.F, z; k′=4, K.S, K.dS),
                        krylov_dim=60, periods=(0.5period(ks22), 2period(ks22)))
    w   = copy(z)
    K₂  = ks(; ℓ=22.1)
    nkroot!(Residual(K₂.G; K₂.S), Jacobian(K₂.L, K₂.F, w; K₂.S, K₂.dS), w; krylov_dim=60,
            krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false)
    @test out.status === :target
    @test norm(Residual(K₂.G; K₂.S)(similar(z), z)) < 1e-9
    @test abs(period(z) - period(w)) < 1e-7
    @test abs(shift(z, 1) - shift(w, 1)) < 1e-7
end


@testset "segments of unequal length               " begin
    # ---- the orbit AB cut at the fractions w of its period: the seeds along it, one state at a time ----
    T = period(ab)
    w = [0.1, 0.4, 0.2, 0.3]
    u = [ab.X[:, 1]]
    for i in 1:3
        push!(u, advance!(_rhs(Lz.G), copy(u[end]), w[i]*T, Lz.G.dt))
    end
    z = ShootingVector(Sliced(u), [T]; w=w)

    # ---- the lengths, kept by every vector made from z, and the checks of the fractions ----
    @test times(z) ≈ w .* T
    @test times(ab) == T/4
    @test similar(z).w === z.w && copy(z).w === z.w && (z .+ z).w === z.w
    @test_throws ArgumentError ShootingVector(Sliced(u), [T]; w=w[1:3])
    @test_throws ArgumentError ShootingVector(Sliced(u), [T]; w=2 .* w)

    # ---- the residual vanishes on the orbit cut so, and the search from it perturbed closes it at the
    #      same period ----
    @test norm(R(similar(z), z)) < 1e-8
    Random.seed!(3)
    y = ShootingVector(Sliced([x .+ 1e-3 .* randn(3) for x in u]), [T]; w=w)
    @test nkroot!(R, Jacobian(Lz.L, Lz.F, y; Lz.dG), y; krylov_dim=20, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
                  maxiter=50, verbose=false).status === :converged
    @test period(y) ≈ T rtol=1e-8

    # ---- the Jacobian off the orbit against central differences of the residual, the seeds and the
    #      period, whose column is wᵢ times the derivative in τᵢ ----
    flat(v)   = [reduce(vcat, v.X); v.d]
    unflat(x) = ShootingVector(Sliced([x[3(i - 1) .+ (1:3)] for i in 1:4]), x[13:13]; w=w)
    Random.seed!(4)
    p = ShootingVector(Sliced([x .+ 1e-2 .* randn(3) for x in u]), [T]; w=w)
    J = Jacobian(Lz.L, Lz.F, p; Lz.dG)
    e(k) = unflat([1.0*(l == k) for l in 1:13])
    A = reduce(hcat, [flat(J*e(k)) for k in 1:13])
    D = reduce(hcat, [(flat(R(similar(p), p .+ 1e-6 .* e(k))) .- flat(R(similar(p), p .- 1e-6 .* e(k)))) ./ 2e-6
                      for k in 1:13])
    @test norm(A[1:12, :] - D[1:12, :]) < 1e-6*norm(A[1:12, :])

    # ---- the coarse matrix of the preconditioner is Wᵀ A W, the period column with the fractions ----
    Random.seed!(1)
    M  = Preconditioner(Lz.L, Lz.F, p; k′=2)
    Jf = Jacobian(Lz.L, Lz.F, p)
    W  = [unflat([zeros(3(i - 1)); M.S.U[a][i]; zeros(3(4 - i)); 0.0]) for i in 1:4 for a in 1:2]
    push!(W, unflat([zeros(12); 1.0]))
    E  = reduce(hcat, [MultipleShooting._restrict(M, Jf*v) for v in W])
    @test norm(Matrix(M.E) - E) < 1e-12*norm(E)
end


@testset "inner product of the states               " begin
    Random.seed!(4)
    flat(v) = [vec(v.X); v.d]

    # ---- complex states: the real part of the Euclidean product, with both batches ----
    z = ShootingVector(randn(ComplexF64, 4, 3), [1.5])
    w = ShootingVector(randn(ComplexF64, 4, 3), [0.5])
    @test dot(z, w) isa Float64
    @test dot(z, w) ≈ real(dot(vec(z.X), vec(w.X))) + 0.75
    @test dot(sliced(z), sliced(w)) ≈ dot(z, w)

    # ---- a product of its own, that of the seeds: a Sliced with a weight on every coordinate,
    #      kept by the vectors made from it ----
    q    = [1.0, 2.0, 3.0]
    wdot = (x, y) -> sum(q .* x .* y)
    z = ShootingVector(Sliced(collect.(eachcol(guess.X)); dot=wdot), copy(guess.d))
    w = similar(z)
    w .= 2 .* z
    @test dot(z, w) ≈ 2*(sum(q .* guess.X.^2) + period(guess)^2)
    @test dot(copy(z), z .+ w) ≈ 1.5*dot(z, w)
    @test norm(z) ≈ sqrt(dot(z, z))

    # ---- the search of AB with it: the same orbit, and a phase condition in the same product ----
    @test search!(z).status === :converged
    @test abs(period(z) - period(ab)) < 1e-8
    v = ShootingVector(Sliced(collect.(eachcol(randn(3, 4))); dot=wdot), [0.3])
    f = lorenz_rhs!(zeros(3), z.X[1], Lz.F.ρ)
    @test period(Jacobian(Lz.L, Lz.F, z; Lz.dG)*v) ≈ sum(q .* f .* v.X[1])
    @test period(JacobianFD(Lz.G, Lz.F, z)*v) ≈ sum(q .* f .* v.X[1])

    # ---- the bases of the subspace are orthonormal in it ----
    M = Preconditioner(Lz.L, Lz.F, z; k′=2)
    @test all(a -> all(dots(M.S.U[a], M.S.U[a]) .≈ 1), 1:2)
    @test maximum(abs, dots(M.S.U[1], M.S.U[2])) < 1e-12
end
