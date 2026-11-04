# A relative periodic orbit known in closed form (the test of NKSearch.jl and MultipleShooting.jl).
# The state is (x, y, u, v). The pair (u, v) is a Hopf oscillator with a limit cycle of radius √μ
# and frequency Ω = ω + βμ; the point (x, y) relaxes to the unit circle and turns at the rate
# c + κ(u² + v²). The rotation of (x, y) is the continuous symmetry. On the orbit
#
#     T = 2π/Ω,   s = -(c + κμ) T.
#
# The propagator advances one state at a time by RK4 with a fixed number of steps, the tangent by
# the linearisation of the scheme, in both layouts: the columns of a matrix, or a vector of states.
using LinearAlgebra

const μ_rpo = 1.0
const ω_rpo = 1.0
const β_rpo = 0.5
const c_rpo = 0.3
const κ_rpo = 0.2

# the right-hand side
function rpo_rhs!(dX, X)
    x, y, u, v = X
    R2 = u^2 + v^2
    r2 = x^2 + y^2
    dX[1] = (1 - r2)*x - (c_rpo + κ_rpo*R2)*y
    dX[2] = (1 - r2)*y + (c_rpo + κ_rpo*R2)*x
    dX[3] = (μ_rpo - R2)*u - (ω_rpo + β_rpo*R2)*v
    dX[4] = (μ_rpo - R2)*v + (ω_rpo + β_rpo*R2)*u
    return dX
end

# its Jacobian
function rpo_jacobian(X)
    x, y, u, v = X
    R2 = u^2 + v^2
    r2 = x^2 + y^2
    a  = c_rpo + κ_rpo*R2
    b  = ω_rpo + β_rpo*R2
    return [1 - r2 - 2x^2   -2x*y - a        -2κ_rpo*u*y                      -2κ_rpo*v*y;
            -2x*y + a       1 - r2 - 2y^2     2κ_rpo*u*x                       2κ_rpo*v*x;
            0.0             0.0               μ_rpo - R2 - 2u^2 - 2β_rpo*u*v   -2u*v - b - 2β_rpo*v^2;
            0.0             0.0              -2u*v + b + 2β_rpo*u^2            μ_rpo - R2 - 2v^2 + 2β_rpo*u*v]
end

# the rotation of (x, y) by s, in place, and its generator
function rpo_rotate!(X, s)
    x    = X[1]
    y    = X[2]
    X[1] = cos(s)*x - sin(s)*y
    X[2] = sin(s)*x + cos(s)*y
    return X
end

rpo_generator!(G, X) = (G .= (-X[2], X[1], 0.0, 0.0); G)

# the orbit on 2 segments, as a matrix of seeds
function rpo_orbit()
    Ω = ω_rpo + β_rpo*μ_rpo
    T = 2π/Ω
    s = -(c_rpo + κ_rpo*μ_rpo)*T
    θ = (c_rpo + κ_rpo*μ_rpo)*T/2
    X = [1.0         cos(θ);
         0.0         sin(θ);
         sqrt(μ_rpo) sqrt(μ_rpo)*cos(Ω*T/2);
         0.0         sqrt(μ_rpo)*sin(Ω*T/2)]
    return (; X, T, s)
end

rpo_rk4!(x, v, τ, dt) = rk4!(rpo_rhs!, rpo_jacobian, x, v, τ, dt)

# The propagator of the tests, with the interface of MultipleShootingV2 in both layouts
struct RPOPropagator
    dt::Float64 # the largest time step
end

# the states of a batch, one at a time: the columns of a matrix, or the states of a vector
rpo_states(X::AbstractMatrix)                 = eachcol(X)
rpo_states(X::AbstractVector{<:AbstractVector}) = X

function MultipleShootingV2.propagate!(P::RPOPropagator, X, τ::Real)
    foreach(x -> rpo_rk4!(x, nothing, τ, P.dt), rpo_states(X))
    return X
end

function MultipleShootingV2.tangent!(P::RPOPropagator, X, V, τ::Real)
    foreach((x, v) -> rpo_rk4!(x, v, τ, P.dt), rpo_states(X), rpo_states(V))
    return V
end

function MultipleShootingV2.rhs!(P::RPOPropagator, F, X)
    foreach(rpo_rhs!, rpo_states(F), rpo_states(X))
    return F
end

function MultipleShootingV2.shift!(P::RPOPropagator, X, s)
    foreach(x -> rpo_rotate!(x, s[1]), rpo_states(X))
    return X
end

function MultipleShootingV2.dshift!(P::RPOPropagator, G, X, j::Int)
    foreach(rpo_generator!, rpo_states(G), rpo_states(X))
    return G
end
