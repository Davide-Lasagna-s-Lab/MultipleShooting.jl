# Kuramoto-Sivashinsky u_t = -u u_x - u_xx - u_xxxx on [0, L) in the odd
# subspace u(-x) = -u(x), u = Σ_{k=1}^{K} a_k sin(κ_k x), κ_k = 2πk/L.
# The state is the vector of the K sine coefficients. The linear part is
# diagonal and treated implicitly; the advection term is evaluated on a grid of
# NX points (NX ≥ 3K + 1 removes aliasing of the quadratic term).
using Flows: flow, couple, CB3R2R3e, TimeStepConstant
using LinearAlgebra

struct KS
    κ::Vector{Float64}
    S::Matrix{Float64}      # NX × K, sine synthesis
    C::Matrix{Float64}      # K × NX, cosine analysis
    u::Vector{Float64}      # grid buffers
    v::Vector{Float64}
    w::Vector{Float64}
    c::Vector{Float64}      # coefficient buffer
end

function KS(L::Float64, K::Int, NX::Int)
    κ = [2π*k/L for k in 1:K]
    x = [L*j/NX for j in 0:NX-1]
    S = [sin(κ[k]*x[j]) for j in 1:NX, k in 1:K]
    C = [2/NX*cos(κ[k]*x[j]) for k in 1:K, j in 1:NX]
    KS(κ, S, C, zeros(NX), zeros(NX), zeros(NX), zeros(K))
end

# linear operator -∂xx - ∂xxxx in the sine basis
linear(ks::KS) = Diagonal(ks.κ.^2 .- ks.κ.^4)

# nonlinear term -(u²/2)_x: u² is even, its cosine coefficients c_k give -(c_k cos)'/2 = κ_k c_k/2 sin
function (ks::KS)(t, a, dadt)
    mul!(ks.u, ks.S, a)
    ks.w .= ks.u.^2
    mul!(ks.c, ks.C, ks.w)
    dadt .= 0.5.*ks.κ.*ks.c
    return dadt
end

# linearised nonlinear term -(u v)_x
struct KSTangent
    ks::KS
end

function (kt::KSTangent)(t, a, dadt, b, dbdt)
    ks = kt.ks
    mul!(ks.u, ks.S, a)
    mul!(ks.v, ks.S, b)
    ks.w .= ks.u.*ks.v
    mul!(ks.c, ks.C, ks.w)
    dbdt .= ks.κ.*ks.c
    return dbdt
end

# right-hand side of the full equation, F(out, a), for the phase condition
struct KSRhs
    ks::KS
    A::Diagonal{Float64, Vector{Float64}}
end
(f::KSRhs)(out, a) = (f.ks(0.0, a, out); out .+= f.A.diag.*a; out)

# Nonlinear flow G, coupled tangent flow L and right-hand side F
function ks_flows(; L=39.0, K=32, NX=128, dt=0.025)
    ks = KS(L, K, NX)
    A = linear(ks)
    G = flow(ks, A, CB3R2R3e(zeros(K)), TimeStepConstant(dt))
    Lt = flow(couple(ks, KSTangent(ks)), couple(A, A),
              CB3R2R3e(couple(zeros(K), zeros(K))), TimeStepConstant(dt))
    return G, Lt, KSRhs(ks, A)
end

# The KS propagator of the tests: the interface of MultipleShootingV2 with the flows of Flows.jl,
# on batches in both layouts, the columns of a matrix or a vector of states, one state at a time
struct KSPropagator{G, L, F}
    G::G # the nonlinear flow, in place on one state
    L::L # the flow of a state coupled to a tangent vector
    f::F # the right-hand side at one state
end

function MultipleShootingV2.propagate!(P::KSPropagator, X::AbstractMatrix, τ::Real)
    # ---- one column at a time, copied, since the flows work on vectors ----
    for j in axes(X, 2)
        x = X[:, j]
        P.G(x, (0, τ))
        X[:, j] .= x
    end

    return X
end

function MultipleShootingV2.rhs!(P::KSPropagator, F::AbstractMatrix, X::AbstractMatrix)
    for j in axes(X, 2)
        F[:, j] .= P.f(similar(X, size(X, 1)), X[:, j])
    end

    return F
end

function MultipleShootingV2.tangent!(P::KSPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real)
    # ---- one pair of columns at a time, copied ----
    for j in axes(X, 2)
        x = X[:, j]
        v = V[:, j]
        P.L(couple(x, v), (0, τ))
        X[:, j] .= x
        V[:, j] .= v
    end

    return V
end

# ---- the sliced layout: a vector of states, each advanced in place ----
function MultipleShootingV2.propagate!(P::KSPropagator, X::AbstractVector{<:AbstractVector}, τ::Real)
    for x in X
        P.G(x, (0, τ))
    end

    return X
end

function MultipleShootingV2.rhs!(P::KSPropagator, F::AbstractVector{<:AbstractVector}, X::AbstractVector{<:AbstractVector})
    for (f, x) in zip(F, X)
        P.f(f, x)
    end

    return F
end

function MultipleShootingV2.tangent!(P::KSPropagator, X::AbstractVector{<:AbstractVector}, V::AbstractVector{<:AbstractVector}, τ::Real)
    for (x, v) in zip(X, V)
        P.L(couple(x, v), (0, τ))
    end

    return V
end
