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

# the tangent L of a vector V of batches of perturbations of the states X, the directions of a
# subspace step, for a tangent whose scheme pairs a state with one perturbation: each batch along a
# copy of the states, and the last along the states themselves, which it advances
function directions!(L, X, V::AbstractVector, τ)
    for a in eachindex(V)
        L(a == lastindex(V) ? X : copy(X), V[a], τ)
    end

    return V
end

# The objects of the dynamics of the Lorenz system at ρ, each a Lorenz{R} of its role R: the flow
# :G, the tangent :L, the right-hand side :F, the tangent forced by ∂f/∂ρ :Lp, and the derivative
# of the flow in τ :dG, each called as MultipleShooting.jl calls it, on a Batched, the columns of a
# matrix, all the states at once, or on a Sliced, one state at a time.
struct Lorenz{R}
    dt::Float64 # the largest time step
     ρ::Float64 # the parameter ρ
end

# all of them at ρ, in a named tuple
lorenz(ρ; dt=1e-3) = (G  = Lorenz{:G}(dt, ρ),
                      L  = Lorenz{:L}(dt, ρ),
                      F  = Lorenz{:F}(dt, ρ),
                      Lp = Lorenz{:Lp}(dt, ρ),
                      dG = Lorenz{:dG}(dt, ρ))

# one of them at another ρ, the family of the searches with a parameter
at(x::Lorenz{R}, ρ) where {R} = Lorenz{R}(x.dt, ρ)

# the right-hand side, its linearisation, and the same forced by ∂f/∂ρ, at the ρ of x, as a flow
# of Flows.jl calls them: the term of the vectors gets the states and their time derivative first
_rhs(x::Lorenz)    = (t, X, dX) -> lorenz_rhs!(dX, X, x.ρ)
_lin(x::Lorenz)    = (t, X, dX, V, dV) -> lorenz_lin!(dV, X, V, x.ρ)
_forced(x::Lorenz) = (t, X, dX, V, dV) -> lorenz_forced!(dV, X, V, x.ρ)

# the linearisation forced by the right-hand side over the time τ, for the derivative in τ
_stretched(x::Lorenz, τ) =
    (t, X, dX, V, dV) -> (lorenz_lin!(dV, X, V, x.ρ); dV .+= dX ./ τ; dV)

# ---- a Batched, the columns of a matrix: all the states at once ----
(G::Lorenz{:G})(X::Batched, τ::Real) = advance!(_rhs(G), X, τ, G.dt)

(L::Lorenz{:L})(X::Batched, V::Batched, τ::Real) = advance!(_rhs(L), _lin(L), X, V, τ, L.dt)

(L::Lorenz{:L})(X::Batched, V::AbstractVector{<:Batched}, τ::Real) = directions!(L, X, V, τ)

(Lp::Lorenz{:Lp})(X::Batched, V::Batched, τ::Real) = advance!(_rhs(Lp), _forced(Lp), X, V, τ, Lp.dt)

(dG::Lorenz{:dG})(X::Batched, V::Batched, τ::Real) =
    (V .= 0; advance!(_rhs(dG), _stretched(dG, τ), X, V, τ, dG.dt))

(F::Lorenz{:F})(dX::Batched, X::Batched) = lorenz_rhs!(dX, X, F.ρ)

# ---- a Sliced: one state at a time ----
(G::Lorenz{:G})(X::Sliced, τ::Real) = (foreach(x -> advance!(_rhs(G), x, τ, G.dt), X); X)

(L::Lorenz{:L})(X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> advance!(_rhs(L), _lin(L), x, v, τ, L.dt), X, V); V)

(L::Lorenz{:L})(X::Sliced, V::AbstractVector{<:Sliced}, τ::Real) = directions!(L, X, V, τ)

(dG::Lorenz{:dG})(X::Sliced, V::Sliced, τ::Real) =
    (foreach((x, v) -> (v .= 0; advance!(_rhs(dG), _stretched(dG, τ), x, v, τ, dG.dt)), X, V); V)

(F::Lorenz{:F})(dX::Sliced, X::Sliced) = (foreach((f, x) -> lorenz_rhs!(f, x, F.ρ), dX, X); dX)

# ---- a Sliced with a time per state, the segments of unequal lengths ----
(G::Lorenz{:G})(X::Sliced, τ::AbstractVector) = (foreach((x, t) -> advance!(_rhs(G), x, t, G.dt), X, τ); X)

(L::Lorenz{:L})(X::Sliced, V::Sliced, τ::AbstractVector) =
    (foreach((x, v, t) -> advance!(_rhs(L), _lin(L), x, v, t, L.dt), X, V, τ); V)

(L::Lorenz{:L})(X::Sliced, V::AbstractVector{<:Sliced}, τ::AbstractVector) = directions!(L, X, V, τ)

(dG::Lorenz{:dG})(X::Sliced, V::Sliced, τ::AbstractVector) =
    (foreach((x, v, t) -> (v .= 0; advance!(_rhs(dG), _stretched(dG, t), x, v, t, dG.dt)), X, V, τ); V)
