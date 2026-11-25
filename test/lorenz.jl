# The Lorenz system at the classical parameters σ = 10, ρ = 28, β = 8/3, advanced by the RK4 of
# Flows.jl, the tangent by the linearisation of the scheme. ρ is a field of the propagator, the parameter of the
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

# The states X advanced over τ by the RK4 of Flows.jl, in equal steps of at most dt: alone, by the
# right-hand side rhs!(t, X, dX), or with the vectors V, by the term vec!(t, X, dX, V, dV) of the
# vectors along the stages of the states, their linearisation or the same with a forcing, so that
# the vectors follow the derivative of the scheme. X is one state or a batch of them, of any array
# type, or a view of one: the flow works on copies, of the type of its registers.
import Flows

function advance!(rhs!, X, τ, dt)
    Y = copy(X)
    ϕ = Flows.flow(rhs!, Flows.RK4(Y), Flows.TimeStepConstant(τ/max(1, ceil(Int, τ/dt))))
    ϕ(Y, (0.0, τ))
    X .= Y

    return X
end

function advance!(rhs!, vec!, X, V, τ, dt)
    Z = Flows.couple(copy(X), copy(V))
    ϕ = Flows.flow(Flows.couple(rhs!, vec!), Flows.RK4(Z), Flows.TimeStepConstant(τ/max(1, ceil(Int, τ/dt))))
    ϕ(Z, (0.0, τ))
    X .= Z[1]
    V .= Z[2]

    return V
end

# tangent! for a vector V of batches of perturbations of the states X, the directions of a subspace
# step, for a propagator whose scheme pairs a state with one perturbation: each batch along a copy
# of the states, and the last along the states themselves, which it advances
function directions!(P, X, V::AbstractVector, τ)
    for a in eachindex(V)
        MultipleShooting.tangent!(P, a == lastindex(V) ? X : copy(X), V[a], τ)
    end

    return V
end

struct LorenzPropagator
    dt::Float64 # the largest time step
     ρ::Float64 # the parameter ρ
end

# the right-hand side, its linearisation, and the same forced by ∂f/∂ρ, at the ρ of P, as a flow
# of Flows.jl calls them: the term of the vectors gets the states and their time derivative first
_rhs(P::LorenzPropagator)    = (t, X, dX) -> lorenz_rhs!(dX, X, P.ρ)
_lin(P::LorenzPropagator)    = (t, X, dX, V, dV) -> lorenz_lin!(dV, X, V, P.ρ)
_forced(P::LorenzPropagator) = (t, X, dX, V, dV) -> lorenz_forced!(dV, X, V, P.ρ)

# the linearisation forced by the right-hand side over the time τ, for the derivative in τ
_stretched(P::LorenzPropagator, τ) =
    (t, X, dX, V, dV) -> (lorenz_lin!(dV, X, V, P.ρ); dV .+= dX ./ τ; dV)

# ---- a Batched, the columns of a matrix: all the states at once ----
MultipleShooting.propagate!(P::LorenzPropagator, X::Batched, τ::Real) =
    (advance!(_rhs(P), X, τ, P.dt); X)

MultipleShooting.tangent!(P::LorenzPropagator, X::Batched, V::Batched, τ::Real) =
    (advance!(_rhs(P), _lin(P), X, V, τ, P.dt); V)

MultipleShooting.tangent!(P::LorenzPropagator, X::Batched, V::AbstractVector{<:Batched}, τ::Real) =
    directions!(P, X, V, τ)

MultipleShooting.forced!(P::LorenzPropagator, X::Batched, V::Batched, τ::Real) =
    (advance!(_rhs(P), _forced(P), X, V, τ, P.dt); V)

MultipleShooting.dpropagate!(P::LorenzPropagator, X::Batched, V::Batched, τ::Real) =
    (V .= 0; advance!(_rhs(P), _stretched(P, τ), X, V, τ, P.dt); V)

MultipleShooting.rhs!(P::LorenzPropagator, F::Batched, X::Batched) = lorenz_rhs!(F, X, P.ρ)

# ---- a Sliced: one state at a time ----
MultipleShooting.propagate!(P::LorenzPropagator, X::Sliced, τ::Real) =
    (foreach(x -> advance!(_rhs(P), x, τ, P.dt), X); X)

MultipleShooting.tangent!(P::LorenzPropagator, X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> advance!(_rhs(P), _lin(P), x, v, τ, P.dt), X, V); V)

MultipleShooting.tangent!(P::LorenzPropagator, X::Sliced, V::AbstractVector{<:Sliced}, τ::Real) =
    directions!(P, X, V, τ)

MultipleShooting.dpropagate!(P::LorenzPropagator, X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> (v .= 0; advance!(_rhs(P), _stretched(P, τ), x, v, τ, P.dt)), X, V); V)

MultipleShooting.rhs!(P::LorenzPropagator, F::Sliced, X::Sliced) =
    (foreach((f, x) -> lorenz_rhs!(f, x, P.ρ), F, X); F)
