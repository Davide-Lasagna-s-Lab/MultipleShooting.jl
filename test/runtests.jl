using MultipleShooting
using NKRoots: nkroot!, gram
import NKRoots
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "sliced.jl"))
include(joinpath(@__DIR__, "lorenz.jl"))
include(joinpath(@__DIR__, "cycles.jl"))
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

# the same vector with a batch that is not an array, the Sliced of sliced.jl, a vector of states
sliced(z) = MultiVector(Sliced(collect.(eachcol(z.X))), copy(z.d))

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
    slice(X) = Sliced(collect.(eachcol(X)))
    flat(v)   = [vec(v.X isa Sliced ? reduce(hcat, v.X) : v.X); v.d]

    # ---- the same vectors with both batches: the columns of a matrix, or a vector of states ----
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

    # ---- the same product with the batch of states ----
    v = MultiVector(randn(3, 4), [0.3])
    @test reduce(hcat, (Jacobian(L, sliced(w))*sliced(v)).X) ≈ (J*v).X
end


@testset "period column                            " begin
    # ---- with a step of 10⁻², the column of the period against central differences of the residual
    #      in T at the same number of steps: dpropagate! of the Lorenz propagator, the forced
    #      tangent, is the derivative of the integrator, the default at the end points only up to
    #      the error of the scheme ----
    Q = LorenzPropagator(1e-2, 28.0)
    Random.seed!(1)
    w  = MultiVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    eT = MultiVector(zeros(3, 4), [1.0])
    flat(v)     = [vec(v.X); v.d]
    residual(y) = flat(Residual(Q)(similar(y), y))
    D = (residual(w .+ 1e-6 .* eT) .- residual(w .- 1e-6 .* eT)) ./ 2e-6
    @test norm(flat(Jacobian(Q, w)*eT) - D) < 1e-8*norm(D)

    X = copy(w.X)
    V = invoke(dpropagate!, Tuple{Any, Any, Any, Real}, Q, X, similar(X), period(w)/4)
    @test norm(vec(circshift(V, (0, 1)))/4 - D[1:12]) > 1e-6*norm(D)

    # ---- the same with the batch of states ----
    @test reduce(hcat, (Jacobian(Q, sliced(w))*sliced(eT)).X) ≈ (Jacobian(Q, w)*eT).X
end


@testset "subspace iteration                       " begin
    # ---- on the orbit with k′ = 2, with both batches from the same random bases ----
    Random.seed!(2)
    S  = Subspace(L, ab; k′=2)
    Sₛ = Subspace(L, sliced(ab); k′=2)
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
    f = lorenz_rhs!(similar(ab.X), ab.X, L.ρ)
    @test maximum(norm(f[:, i] - Vᵢ(i)*(Vᵢ(i)'*f[:, i]))/norm(f[:, i]) for i in 1:4) < 1e-6

    # ---- the two layouts agree ----
    @test all(reduce(hcat, Sₛ.V[a]) ≈ S.V[a] for a in 1:2)
    @test Sₛ.E ≈ S.E

    # ---- a step leaves the end points of the segments, bases that span the images that tangent!
    #      gives of each direction alone, at the point they reach, and the Galerkin blocks of the
    #      bases before it, Eᵢ = Uᵢ₊₁ᵀ ψᵢ Uᵢ ----
    τ = period(ab)/4
    W = map(S.V) do Vᵃ
        tangent!(L, copy(ab.X), copy(Vᵃ), τ)
    end
    Wᵢ(i) = reduce(hcat, [Wᵇ[:, i] for Wᵇ in W])
    iterate!(S)
    @test all(Vᵢ(i % 4 + 1)*(Vᵢ(i % 4 + 1)'*Wᵢ(i)) ≈ Wᵢ(i) for i in 1:4)
    @test S.tmp ≈ propagate!(L, copy(ab.X), τ)
    @test all(S.E[:, :, i] ≈ Uᵢ(i % 4 + 1)'*reduce(hcat, [Wᵇ[:, i] for Wᵇ in W]) for i in 1:4)

    # ---- tangent! advances the batches of several directions along the states, each as on the
    #      batch alone, and leaves the end points in the states ----
    X = copy(ab.X)
    V = copy.(S.V)
    W = [tangent!(L, copy(ab.X), copy(Vᵃ), τ) for Vᵃ in S.V]
    tangent!(L, X, V, τ)
    @test X ≈ propagate!(L, copy(ab.X), τ)
    @test all(V[a] ≈ W[a] for a in 1:2)

    # ---- on bases that have settled, the update passes the test after one step ----
    MultipleShooting.update!(S, ab; tol=1e-12)
    @test S.steps == 1
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
    S = Subspace(K, ks22; k′=4)
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
    g = dshift!(K, similar(ks22.X), ks22.X, 1)
    Vᵢ(i) = reduce(hcat, [Vᵃ[:, i] for Vᵃ in S.V])
    @test maximum(norm(g[:, i] - Vᵢ(i)*(Vᵢ(i)'*g[:, i]))/norm(g[:, i]) for i in 1:4) < 1e-10
end


@testset "subspace update to a tolerance           " begin
    # ---- steps until the turn of the first three directions passes the test, and no more than
    #      the cap ----
    Random.seed!(2)
    S = Subspace(K, ks22; k′=4)
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
    J = Jacobian(L, ab)
    M = Preconditioner(L, ab; k′=2)
    W = MultiVector[]
    for i in 1:4, a in 1:2
        w = MultiVector(zeros(3, 4), [0.0])
        w.X[:, i] .= M.S.U[a][:, i]
        push!(W, w)
    end
    push!(W, MultiVector(zeros(3, 4), [1.0]))
    E = reduce(hcat, [MultipleShooting._restrict(M, J*w) for w in W])
    @test norm(Matrix(M.E)[:, 1:8] - E[:, 1:8]) < 1e-12*norm(E)

    # ---- the column of the period of E is f at the end points, that of the Jacobian the
    #      derivative of the integrator: they differ by the error of the scheme ----
    @test norm(Matrix(M.E)[:, 9] - E[:, 9]) < 1e-6*norm(E)

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

    # ---- a solve with the coarse matrix whose relative residual is above the tolerance is an
    #      error: here the factors are those of half the matrix, and the residual is one ----
    B = Preconditioner(L, ab; k′=2)
    B.E.nzval .*= 2
    @test_throws ErrorException ldiv!(similar(ab), B, unflat(x))
    @test_throws ErrorException gram(B, Q)

    # ---- the KS search with the preconditioner, k′ = 4, reaches the orbit ----
    Random.seed!(2)
    z  = copy(ks22_guess)
    Jk = Jacobian(K, z)
    @test nkroot!(Residual(K), Jk, z; precond=Preconditioner(K, z; k′=4), flexible=true, krylov_dim=60,
                  krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false).status === :converged
end


@testset "preconditioner: steps per update, reuse   " begin
    # ---- five sweeps and one step when built, then a fixed number of steps per update ----
    Random.seed!(2)
    z = copy(ks22_guess)
    M = Preconditioner(K, z; k′=4, sweeps=5, tol=0.0, maxsteps=1, cap=:keep)
    @test M.S.steps == 1
    NKRoots.update!(M, z)
    @test M.S.steps == 1 && M.updated == [true]

    # ---- at a fixed point, the steps to a tolerance on the leading directions stop once their
    #      turn passes the test ----
    R = Preconditioner(K, z; k′=4, k=3, sweeps=5, tol=1e-6, maxsteps=40, cap=:keep)
    NKRoots.update!(R, z)
    @test maxturn(R.S; k=3) <= 1e-6 && R.S.steps < 40

    # ---- every = 2: updates at the Newton steps 1 and 3, and at step 2 the same E with no step ----
    U = Preconditioner(K, z; k′=4, sweeps=5, tol=0.0, maxsteps=1, cap=:keep, every=2)
    NKRoots.update!(U, z)
    E = copy(U.E)
    w = copy(z)
    w.X .+= 1e-3 .* randn(size(w.X))
    NKRoots.update!(U, w)
    @test U.E == E && U.S.z === z
    NKRoots.update!(U, w)
    @test U.updated == [true, false, true] && U.S.z === w && U.E != E

    # ---- every = 0: the first Newton step only ----
    Z = Preconditioner(K, z; k′=4, sweeps=5, tol=0.0, maxsteps=1, cap=:keep, every=0)
    for _ in 1:3
        NKRoots.update!(Z, w)
    end
    @test Z.updated == [true, false, false]
    @test_throws ArgumentError Preconditioner(K, z; k′=4, every=-1)
    @test_throws ArgumentError Preconditioner(K, z; k′=4, k=5)

    # ---- the size of the bases has no default: a call with k alone fails, not read as k′ ----
    @test_throws UndefKeywordError Preconditioner(K, z; k=4)
    @test_throws UndefKeywordError Subspace(K, z; k=4)

    # ---- the KS search with two steps per update and with the preconditioner of every second step
    #      both reach the orbit ----
    for (every, maxsteps) in ((1, 2), (2, 1))
        Random.seed!(2)
        u = copy(ks22_guess)
        P = Preconditioner(K, u; k′=4, sweeps=5, tol=0.0, maxsteps=maxsteps, cap=:keep, every=every)
        @test nkroot!(Residual(K), Jacobian(K, u), u; precond=P, flexible=true, krylov_dim=60,
                      krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=40, verbose=false).status === :converged
    end
end


@testset "Jacobian by finite differences           " begin
    # ---- off the orbit, its products against those of the tangent equation, with both batches ----
    Random.seed!(1)
    w = MultiVector(ab.X .+ 1e-2 .* randn(3, 4), copy(ab.d))
    v = MultiVector(randn(3, 4), [0.3])
    flat(u) = [vec(u.X); u.d]
    J = JacobianFD(L, w)
    @test norm(flat(J*v) - flat(Jacobian(L, w)*v)) < 1e-6*norm(flat(Jacobian(L, w)*v))
    @test norm(flat(JacobianFD(L, w; difference=:central)*v) - flat(Jacobian(L, w)*v)) <
          1e-8*norm(flat(Jacobian(L, w)*v))
    @test reduce(hcat, (JacobianFD(L, sliced(w))*sliced(v)).X) ≈ (J*v).X

    # ---- the searches of AB and of the KS orbit, with a shift, find the same orbits ----
    z = copy(guess)
    @test nkroot!(F, JacobianFD(L, z), z; krylov_dim=20, krylov_tol=1e-8, Δ=1.0, tol=1e-10, maxiter=50,
                  verbose=false).status === :converged
    @test abs(period(z) - period(ab)) < 1e-8
    z = copy(ks22_guess)
    @test nkroot!(Residual(K), JacobianFD(K, z), z; krylov_dim=60, krylov_tol=1e-8, Δ=1.0, tol=1e-10,
                  maxiter=40, verbose=false).status === :converged
    @test abs(period(z) - period(ks22)) < 1e-6
    @test abs(shift(z, 1) - shift(ks22, 1)) < 1e-6
end


@testset "bounds on the period                     " begin
    # ---- within the bounds the residual is the same, outside it is infinite, with no integration ----
    @test Residual(L; bounds=(1.0, 2.0))(similar(ab), ab).X == F(similar(ab), ab).X
    @test norm(Residual(L; bounds=(1.0, 1.5))(similar(ab), ab)) == Inf
    @test norm(Residual(L; bounds=(1.6, 2.0))(similar(ab), ab)) == Inf

    # ---- a period that is not positive is out of the default bounds ----
    for T in (0.0, -period(ab))
        z = MultiVector(copy(ab.X), [T])
        @test norm(F(similar(z), z)) == Inf
    end
end


@testset "a parameter as an unknown, two cycles     " begin
    cycles(μ) = CyclePropagator(1e-2, μ)
    outer(μ)  = (1 + sqrt(1 + 4μ))/2
    inner(μ)  = (1 - sqrt(1 + 4μ))/2
    radius²(z) = sum(abs2, z.X[:, 1])

    # ---- below the fold there is no orbit: from the circle of the fold at μ = -0.3 the search at
    #      a fixed parameter does not converge, and the search with the parameter free lands on a
    #      circle at a parameter above the fold ----
    z = circle(0.5)
    @test nkroot!(Residual(cycles(-0.3)), Jacobian(cycles(-0.3), z), z; krylov_dim=9, krylov_tol=1e-10, Δ=0.1,
                  tol=1e-10, maxiter=50, verbose=false).status !== :converged
    z   = circle(0.5)
    out = freesearch!(cycles, z, -0.3; krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :converged
    @test out.p > -0.25
    @test norm(Residual(cycles(out.p))(similar(z), z)) < 1e-10
    @test abs(radius²(z) - outer(out.p)) < 1e-6 || abs(radius²(z) - inner(out.p)) < 1e-6

    # ---- the column of the parameter from the forced tangent equation and from the central
    #      difference of the residual, off an orbit, and the same search with the second ----
    z = circle(0.6)
    R(u, μ) = Residual(cycles(μ))(similar(u), u)
    @test norm(MultipleShooting._forced(cycles(-0.2), z) .- (R(z, -0.2 + 1e-5) .- R(z, -0.2 - 1e-5)) ./ 2e-5) < 1e-8
    z = circle(0.5)
    @test abs(freesearch!(cycles, z, -0.3; column=:difference, krylov_tol=1e-10, krylov_dim=10).p - out.p) < 1e-8
    @test_throws ArgumentError freesearch!(cycles, z, -0.3; column=:central)

    # ---- on an orbit it stays where it is, up to the error of the scheme ----
    z   = circle(outer(-0.1))
    out = freesearch!(cycles, z, -0.1; krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :converged
    @test abs(out.p + 0.1) < 1e-8

    # ---- a trial parameter out of bounds is refused, and the search collapses ----
    z = circle(0.5)
    @test freesearch!(cycles, z, -0.3; bounds=(-0.35, -0.26), krylov_tol=1e-10, krylov_dim=10).status === :collapsed

    # ---- with no target the continuation from the outer circle, the parameter shrinking, ends at
    #      the fold, μ = -1/4 and r² = 1/2 ----
    z   = circle(outer(-0.1))
    out = continuation!(cycles, z, -0.1; sense=-1, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :fold
    @test abs(out.p + 0.25) < 1e-6
    @test abs(radius²(z) - 0.5) < 2e-2
    @test all(<(0), out.path.slope[1:(end - 1)])

    # ---- with a target above it, it passes the fold and ends on the inner circle at the target ----
    z   = circle(outer(-0.1))
    out = continuation!(cycles, z, -0.1; sense=-1, target=-0.05, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :target
    @test abs(out.p + 0.05) < 1e-9
    @test abs(radius²(z) - inner(-0.05)) < 1e-6
    @test minimum(out.path.p) < -0.24

    # ---- and with the parameter growing it ends on the outer circle at the same target ----
    z   = circle(outer(-0.1))
    out = continuation!(cycles, z, -0.1; target=-0.05, krylov_tol=1e-10, krylov_dim=10)
    @test out.status === :target
    @test abs(radius²(z) - outer(-0.05)) < 1e-6
    @test_throws ArgumentError continuation!(cycles, z, -0.05; sense=2)
end


@testset "continuation of AB and of the KS orbit    " begin
    # ---- AB from ρ = 28 to 28.5 against the search at 28.5, and the first slope dT/dρ against
    #      the tangent sensitivity ----
    lorenz(ρ) = LorenzPropagator(1e-3, ρ)
    z   = copy(ab)
    out = continuation!(lorenz, z, 28.0; target=28.5, krylov_tol=1e-10, krylov_dim=20)
    w   = copy(ab)
    nkroot!(Residual(lorenz(28.5)), Jacobian(lorenz(28.5), w), w; krylov_dim=20, krylov_tol=1e-10, Δ=1.0,
            tol=1e-12, maxiter=50, verbose=false)
    @test out.status === :target
    @test abs(period(z) - period(w)) < 1e-8
    @test abs((out.path.period[2] - out.path.period[1])/(out.path.p[2] - out.path.p[1]) -
              period(tangent_sensitivity(L, ab))) < 1e-2*abs(period(tangent_sensitivity(L, ab)))

    # ---- the Newton homotopy from the guess of AB, the amplitude of its residual from 1 to 0 ----
    z   = copy(guess)
    F₀  = F(similar(z), z)
    out = continuation!(a -> L, z, 1.0; sense=-1, target=0.0, krylov_tol=1e-10, krylov_dim=20,
                        residual=(r, u, a) -> (F(r, u); r .= r .- a .* F₀))
    @test out.status === :target
    @test norm(F(similar(z), z)) < 1e-8
    @test abs(period(z) - period(ab)) < 1e-6

    # ---- the KS orbit in the logarithm of the length of the domain, with the preconditioner and
    #      bounds on the period, from L = 22 to 22.1 against the search there. The length is in
    #      the shift too, so that the column is the central difference ----
    Random.seed!(2)
    ks(λ) = KSPropagator(; L=exp(λ))
    z   = copy(ks22)
    out = continuation!(ks, z, log(22.0); target=log(22.1), precond=Preconditioner(K, z; k′=4), krylov_dim=60,
                        periods=(0.5period(ks22), 2period(ks22)), column=:difference)
    w   = copy(z)
    nkroot!(Residual(ks(log(22.1))), Jacobian(ks(log(22.1)), w), w; krylov_dim=60, krylov_tol=1e-8, Δ=1.0,
            tol=1e-10, maxiter=40, verbose=false)
    @test out.status === :target
    @test norm(Residual(ks(log(22.1)))(similar(z), z)) < 1e-9
    @test abs(period(z) - period(w)) < 1e-7
    @test abs(shift(z, 1) - shift(w, 1)) < 1e-7
end


@testset "inner product of the states               " begin
    Random.seed!(4)
    flat(v) = [vec(v.X); v.d]

    # ---- complex states: the real part of the Euclidean product, with both batches ----
    z = MultiVector(randn(ComplexF64, 4, 3), [1.5])
    w = MultiVector(randn(ComplexF64, 4, 3), [0.5])
    @test dot(z, w) isa Float64
    @test dot(z, w) ≈ real(dot(vec(z.X), vec(w.X))) + 0.75
    @test dot(sliced(z), sliced(w)) ≈ dot(z, w)

    # ---- a product of its own, that of the batch: the Sliced of the tests with a weight on every
    #      coordinate, kept by the vectors made from it ----
    q = [1.0, 2.0, 3.0]
    z = MultiVector(Sliced(collect.(eachcol(guess.X)), q), copy(guess.d))
    w = similar(z)
    w .= 2 .* z
    @test dot(z, w) ≈ 2*(sum(q .* guess.X.^2) + period(guess)^2)
    @test dot(copy(z), z .+ w) ≈ 1.5*dot(z, w)
    @test norm(z) ≈ sqrt(dot(z, z))

    # ---- the search of AB with it: the same orbit, and a phase condition in the same product ----
    @test search!(z).status === :converged
    @test abs(period(z) - period(ab)) < 1e-8
    v = MultiVector(Sliced(collect.(eachcol(randn(3, 4))), q), [0.3])
    f = lorenz_rhs!(zeros(3), z.X[1], L.ρ)
    @test period(Jacobian(L, z)*v) ≈ sum(q .* f .* v.X[1])
    @test period(JacobianFD(L, z)*v) ≈ sum(q .* f .* v.X[1])

    # ---- the bases of the subspace are orthonormal in it ----
    M = Preconditioner(L, z; k′=2)
    @test all(a -> all(dots(M.S.U[a], M.S.U[a]) .≈ 1), 1:2)
    @test maximum(abs, dots(M.S.U[1], M.S.U[2])) < 1e-12
end
