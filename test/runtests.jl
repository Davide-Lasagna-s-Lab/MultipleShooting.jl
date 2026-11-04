using MultipleShootingV2
using NKRoots: nkroot!
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "ks.jl"))
include(joinpath(@__DIR__, "rk4.jl"))
include(joinpath(@__DIR__, "rpo.jl"))
include(joinpath(@__DIR__, "lorenz.jl"))


# ---- the KS model of the notes: L = 39, odd subspace, 32 sine modes ----
P = KSPropagator(ks_flows()...)

# ---- the orbit of period 53.134 on its 27 seeds ----
lines = readlines(joinpath(@__DIR__, "data", "T_53.134.txt"))
T0    = parse(Float64, lines[1])
X0    = reduce(hcat, [parse.(Float64, split(l)) for l in lines[2:end]])
N0    = size(X0, 2)


@testset "propagator interface                     " begin
    τ = T0/N0

    # ---- every seed of the orbit lands on the next one ----
    Y = propagate!(P, copy(X0), τ)
    @test norm(Y - circshift(X0, (0, -1))) < 1e-6*norm(X0)

    # ---- the tangent equation against a forward difference ----
    Random.seed!(1)
    V  = randn(size(X0))
    h  = 1e-6
    Xh = propagate!(P, X0 .+ h .* V, τ)
    Xt = copy(X0)
    Vt = tangent!(P, Xt, copy(V), τ)
    @test norm(Vt - (Xh - Y)./h) < 1e-4*norm(Vt)
    @test Xt ≈ Y
end


@testset "multivector                              " begin
    Random.seed!(2)
    A = randn(4, 3)
    B = randn(4, 3)
    sliced(X) = collect.(eachcol(X))
    flat(v)   = [vec(v.X isa Vector{<:Vector} ? reduce(hcat, v.X) : v.X); v.d]

    # ---- the same vectors in both layouts: the columns of a matrix, or a vector of states ----
    for (z, w) in ((MultiVector(copy(A), [1.5]), MultiVector(copy(B), [-0.5])),
                   (MultiVector(sliced(A), [1.5]), MultiVector(sliced(B), [-0.5])))
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


@testset "residual                                 " begin
    # ---- zero on the orbit, with its 27 seeds and with every third of them ----
    for every in (1, 3)
        z = MultiVector(X0[:, 1:every:end], [T0])
        r = Residual(P)(similar(z), z)
        @test norm(r) < 1e-6*norm(z)
        @test r.d == [0.0]
    end

    # ---- not zero off it, with the end of segment i in the row of segment i + 1 ----
    Random.seed!(3)
    z = MultiVector(X0 .+ 1e-3 .* randn(size(X0)), [T0])
    X = copy(z.X)
    r = Residual(P)(similar(z), z)
    Φ = propagate!(P, copy(z.X), T0/N0)
    @test norm(r) > 1e-4*norm(z)
    @test r.X[:, 2] ≈ Φ[:, 1] - z.X[:, 2]
    @test r.X[:, 1] ≈ Φ[:, N0] - z.X[:, 1]
    @test z.X == X

    # ---- the same residual in the sliced layout, and the seeds left as they were ----
    zs = MultiVector(collect.(eachcol(z.X)), [T0])
    Xs = deepcopy(zs.X)
    rs = Residual(P)(similar(zs), zs)
    @test reduce(hcat, rs.X) ≈ r.X
    @test rs.d == [0.0]
    @test zs.X == Xs
end


@testset "Jacobian                                 " begin
    # ---- a point off the orbit, with 3 seeds so that the dense matrices stay small ----
    Random.seed!(4)
    X = X0[:, 1:9:end] .+ 1e-3 .* randn(size(X0, 1), 3)
    z = MultiVector(copy(X), [T0])
    m = nsegments(z)
    n = size(X, 1)
    J = Jacobian(P, z)

    # ---- the dense Jacobian, column by column ----
    unit(k) = (e = zeros(n*m + 1); e[k] = 1; MultiVector(reshape(e[1:n*m], n, m), [e[end]]))
    flat(v) = [vec(v.X); v.d]
    A = reduce(hcat, [flat(J*unit(k)) for k in 1:n*m + 1])

    # ---- the columns of the seeds against central differences of the residual ----
    h = 1e-5
    D = reduce(hcat, [flat(Residual(P)(similar(z), z .+ h .* unit(k)) .-
                           Residual(P)(similar(z), z .- h .* unit(k))) ./ (2h) for k in 1:n*m])
    @test norm(A[1:n*m, 1:n*m] - D[1:n*m, :]) < 1e-7*norm(A[1:n*m, 1:n*m])

    # ---- the period column is f at the end points over m, in the row of the next segment: the
    #      derivative of the exact flow, not of the scheme, so not compared with differences ----
    Φ = propagate!(P, copy(X), T0/m)
    F = rhs!(P, similar(Φ), Φ)
    @test A[1:n*m, end] ≈ vec(circshift(F, (0, 1)))./m

    # ---- the phase row is f at the first seed ----
    f₁ = rhs!(P, zeros(n, 1), X[:, 1:1])
    @test A[end, :] ≈ [vec(f₁); zeros(n*(m - 1) + 1)]

    # ---- the point is left as it was ----
    @test z.X == X

    # ---- after z moves in place, update! ----
    Random.seed!(5)
    v = MultiVector(randn(n, m), [0.3])
    z.X .+= 1e-3 .* randn(n, m)
    z.d .= T0 + 0.01
    update!(J, z)
    w = Jacobian(P, z)*v
    @test norm((J*v).X - w.X) < 1e-10*norm(w.X)
    @test (J*v).d ≈ w.d

    # ---- the same product in the sliced layout ----
    zs = MultiVector(collect.(eachcol(z.X)), copy(z.d))
    vs = MultiVector(collect.(eachcol(v.X)), [0.3])
    ws = Jacobian(P, zs)*vs
    @test reduce(hcat, ws.X) ≈ w.X
    @test ws.d ≈ w.d
end


@testset "relative periodic orbit                  " begin
    o = rpo_orbit()
    Q = RPOPropagator(1e-3)

    # ---- the residual vanishes on the orbit known in closed form ----
    z = MultiVector(copy(o.X), [o.T, o.s])
    @test norm(Residual(Q)(similar(z), z)) < 1e-8

    # ---- off it, the columns of the seeds and of the shift against central differences ----
    Random.seed!(6)
    z = MultiVector(o.X .+ 1e-2 .* randn(4, 2), [o.T, o.s + 0.1])
    J = Jacobian(Q, z)
    unit(k) = (e = zeros(10); e[k] = 1; MultiVector(reshape(e[1:8], 4, 2), e[9:10]))
    flat(v) = [vec(v.X); v.d]
    D(k)    = flat(Residual(Q)(similar(z), z .+ 1e-6 .* unit(k)) .-
                   Residual(Q)(similar(z), z .- 1e-6 .* unit(k))) ./ 2e-6
    for k in [1:8; 10]
        @test norm(flat(J*unit(k))[1:8] - D(k)[1:8]) < 1e-7
    end
end


@testset "Newton–Krylov with NKRoots               " begin
    # ---- the KS orbit on 9 of its seeds, perturbed, reconverged by the hookstep ----
    Random.seed!(7)
    z = MultiVector(X0[:, 1:3:end] .+ 1e-4 .* randn(size(X0, 1), 9), [T0 + 1e-3])
    F = Residual(P)
    J = Jacobian(P, z)
    out = nkroot!(F, J, z; krylov_dim=100, krylov_tol=1e-6, Δ=1.0, tol=1e-9, verbose=false)
    @test out.status === :converged
    @test abs(period(z) - T0) < 1e-5
end


@testset "Lorenz orbit AB from a near recurrence   " begin
    L = LorenzPropagator(1e-3)

    # ---- a trajectory on the attractor, sampled every 0.01 after a transient ----
    x = lorenz_rk4!([1.0, 1.0, 1.0], nothing, 20.0, 1e-3)
    Y = zeros(3, 6000)
    for k in axes(Y, 2)
        Y[:, k] .= x
        lorenz_rk4!(x, nothing, 0.01, 1e-3)
    end

    # ---- its closest return with a period between 1.4 and 1.7, where only AB lives ----
    d, k, p = minimum((norm(Y[:, k + p] - Y[:, k]), k, p) for k in 1:size(Y, 2) - 170, p in 140:170)
    T = 0.01*p

    # ---- four seeds along that stretch, converged by the hookstep ----
    X = reduce(hcat, [lorenz_rk4!(Y[:, k], nothing, (i - 1)*T/4, 1e-3) for i in 1:4])
    z = MultiVector(X, [T])
    out = nkroot!(Residual(L), Jacobian(L, z), z; krylov_dim=20, krylov_tol=1e-8, Δ=1.0,
                  tol=1e-10, maxiter=50, verbose=false)
    @test out.status === :converged
    @test abs(period(z) - 1.5586522107) < 1e-6
end
