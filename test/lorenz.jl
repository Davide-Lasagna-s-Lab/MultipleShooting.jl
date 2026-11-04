# The Lorenz system at the classical parameters σ = 10, ρ = 28, β = 8/3, whose shortest periodic
# orbit, AB, has the period T = 1.5586522… (Viswanath 2003). The propagator advances the columns of
# a matrix one at a time by RK4, the tangent by the linearisation of the scheme.
const σ_lz = 10.0
const ρ_lz = 28.0
const β_lz = 8/3

function lorenz_rhs!(dx, x)
    dx[1] = σ_lz*(x[2] - x[1])
    dx[2] = x[1]*(ρ_lz - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β_lz*x[3]
    return dx
end

lorenz_jacobian(x) = [-σ_lz         σ_lz   0.0;
                      ρ_lz - x[3]   -1.0   -x[1];
                      x[2]          x[1]   -β_lz]

lorenz_rk4!(x, v, τ, dt) = rk4!(lorenz_rhs!, lorenz_jacobian, x, v, τ, dt)

struct LorenzPropagator
    dt::Float64 # the largest time step
end

function MultipleShootingV2.propagate!(P::LorenzPropagator, X::AbstractMatrix, τ::Real)
    foreach(x -> lorenz_rk4!(x, nothing, τ, P.dt), eachcol(X))
    return X
end

function MultipleShootingV2.tangent!(P::LorenzPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real)
    foreach((x, v) -> lorenz_rk4!(x, v, τ, P.dt), eachcol(X), eachcol(V))
    return V
end

function MultipleShootingV2.rhs!(P::LorenzPropagator, F::AbstractMatrix, X::AbstractMatrix)
    foreach(lorenz_rhs!, eachcol(F), eachcol(X))
    return F
end
