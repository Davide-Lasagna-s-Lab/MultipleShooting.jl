# The Lorenz system at the classical parameters σ = 10, ρ = 28, β = 8/3, advanced by RK4, the
# tangent by the linearisation of the scheme. ρ is a field of the propagator, the parameter of the
# sensitivity. The right-hand side and its linearisation are written for one state, a 3-vector, and
# applied column by column to a 3 × B matrix, so that the same RK4 serves both layouts.
const σ_lz = 10.0
const β_lz = 8/3

# ---- the right-hand side at ρ, on one state and on the columns of a matrix ----
function lorenz_rhs!(dx::AbstractVector, x::AbstractVector, ρ)
    dx[1] = σ_lz*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β_lz*x[3]
    return dx
end

lorenz_rhs!(dX::AbstractMatrix, X::AbstractMatrix, ρ) =
    (foreach((dx, x) -> lorenz_rhs!(dx, x, ρ), eachcol(dX), eachcol(X)); dX)

# ---- its linearisation at x applied to v, on one state and on the columns of matrices ----
function lorenz_lin!(dv::AbstractVector, x::AbstractVector, v::AbstractVector, ρ)
    dv[1] = σ_lz*(v[2] - v[1])
    dv[2] = (ρ - x[3])*v[1] - v[2] - x[1]*v[3]
    dv[3] = x[2]*v[1] + x[1]*v[2] - β_lz*v[3]
    return dv
end

lorenz_lin!(dV::AbstractMatrix, X::AbstractMatrix, V::AbstractMatrix, ρ) =
    (foreach((dv, x, v) -> lorenz_lin!(dv, x, v, ρ), eachcol(dV), eachcol(X), eachcol(V)); dV)

# ---- the same forced by the derivative of the right-hand side in ρ, ∂f/∂ρ = (0, x₁, 0) ----
lorenz_forced!(dv::AbstractVector, x::AbstractVector, v::AbstractVector, ρ) =
    (lorenz_lin!(dv, x, v, ρ); dv[2] += x[1]; dv)

lorenz_forced!(dV::AbstractMatrix, X::AbstractMatrix, V::AbstractMatrix, ρ) =
    (foreach((dv, x, v) -> lorenz_forced!(dv, x, v, ρ), eachcol(dV), eachcol(X), eachcol(V)); dV)

# RK4 with a fixed number of steps, over τ in steps of at most dt: the states alone, or the states
# with tangent vectors advanced by the linearisation of the scheme. It needs only similar and
# broadcasting, so that X may be one state or a batch of them, of any array type: rhs!(dX, X) is
# the right-hand side and lin!(dV, X, V) its linearisation at X applied to V, on whatever X is.

# the states X
function rk4!(rhs!, X, τ, dt)
    n = max(1, ceil(Int, τ/dt))
    h = τ/n

    # ---- the stages ----
    K1, K2, K3, K4 = similar(X), similar(X), similar(X), similar(X)
    Y              = similar(X)

    for _ in 1:n
        rhs!(K1, X)
        Y .= X .+ (h/2) .* K1
        rhs!(K2, Y)
        Y .= X .+ (h/2) .* K2
        rhs!(K3, Y)
        Y .= X .+ h .* K3
        rhs!(K4, Y)
        X .+= (h/6) .* (K1 .+ 2 .* K2 .+ 2 .* K3 .+ K4)
    end

    return X
end

# the states X and the tangent vectors V, the stages of V along those of X
function rk4!(rhs!, lin!, X, V, τ, dt)
    n = max(1, ceil(Int, τ/dt))
    h = τ/n

    # ---- the stages of the states, and of the tangent vectors ----
    K1, K2, K3, K4 = similar(X), similar(X), similar(X), similar(X)
    X2, X3, X4     = similar(X), similar(X), similar(X)
    L1, L2, L3, L4 = similar(V), similar(V), similar(V), similar(V)
    W              = similar(V)

    for _ in 1:n
        # ---- the stages of the states ----
        rhs!(K1, X)
        X2 .= X .+ (h/2) .* K1
        rhs!(K2, X2)
        X3 .= X .+ (h/2) .* K2
        rhs!(K3, X3)
        X4 .= X .+ h .* K3
        rhs!(K4, X4)

        # ---- the same stages linearised, along those of the states ----
        lin!(L1, X, V)
        W .= V .+ (h/2) .* L1
        lin!(L2, X2, W)
        W .= V .+ (h/2) .* L2
        lin!(L3, X3, W)
        W .= V .+ h .* L3
        lin!(L4, X4, W)

        V .+= (h/6) .* (L1 .+ 2 .* L2 .+ 2 .* L3 .+ L4)
        X .+= (h/6) .* (K1 .+ 2 .* K2 .+ 2 .* K3 .+ K4)
    end

    return V
end

struct LorenzPropagator
    dt::Float64 # the largest time step
     ρ::Float64 # the parameter ρ
end

# the right-hand side, its linearisation, and the same forced by ∂f/∂ρ, at the ρ of P, as RK4
# calls them
_rhs(P::LorenzPropagator)    = (dX, X) -> lorenz_rhs!(dX, X, P.ρ)
_lin(P::LorenzPropagator)    = (dV, X, V) -> lorenz_lin!(dV, X, V, P.ρ)
_forced(P::LorenzPropagator) = (dV, X, V) -> lorenz_forced!(dV, X, V, P.ρ)

# the linearisation forced by the right-hand side over the time τ, for the derivative in τ
_stretched(P::LorenzPropagator, τ) =
    (dV, X, V) -> (lorenz_lin!(dV, X, V, P.ρ); dV .+= lorenz_rhs!(similar(X), X, P.ρ) ./ τ; dV)

# ---- the columns of a matrix: all the states at once ----
MultipleShooting.propagate!(P::LorenzPropagator, X::AbstractMatrix, τ::Real) =
    (rk4!(_rhs(P), X, τ, P.dt); X)

MultipleShooting.tangent!(P::LorenzPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real) =
    (rk4!(_rhs(P), _lin(P), X, V, τ, P.dt); V)

MultipleShooting.forced!(P::LorenzPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real) =
    (rk4!(_rhs(P), _forced(P), X, V, τ, P.dt); V)

MultipleShooting.dpropagate!(P::LorenzPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real) =
    (V .= 0; rk4!(_rhs(P), _stretched(P, τ), X, V, τ, P.dt); V)

MultipleShooting.rhs!(P::LorenzPropagator, F::AbstractMatrix, X::AbstractMatrix) = lorenz_rhs!(F, X, P.ρ)

# ---- the Sliced of sliced.jl: one state at a time ----
MultipleShooting.propagate!(P::LorenzPropagator, X::Sliced, τ::Real) =
    (foreach(x -> rk4!(_rhs(P), x, τ, P.dt), X); X)

MultipleShooting.tangent!(P::LorenzPropagator, X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> rk4!(_rhs(P), _lin(P), x, v, τ, P.dt), X, V); V)

MultipleShooting.dpropagate!(P::LorenzPropagator, X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> (v .= 0; rk4!(_rhs(P), _stretched(P, τ), x, v, τ, P.dt)), X, V); V)

MultipleShooting.rhs!(P::LorenzPropagator, F::Sliced, X::Sliced) =
    (foreach((f, x) -> lorenz_rhs!(f, x, P.ρ), F, X); F)
