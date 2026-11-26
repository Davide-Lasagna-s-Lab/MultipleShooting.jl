# The Kuramoto–Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx on [0, ℓ), periodic, and the
# objects of its dynamics for MultipleShooting.jl, which the scripts ks.jl and
# ../parameter/parameter.jl include.
#
# The equation is written on x̃ = 2πx/ℓ ∈ [0, 2π), for the Fourier coefficients û_k of the modes
# k = 1, …, 21, the last below N/3 on a grid of N = 64 points,
#
#     u = Σ_{0 < |k| ≤ 21} û_k e^{ikx̃},   û_t = Λ û + N(û),   Λ = κ² - κ⁴,   N(û) = -(iκ/2) (u²)^,
#
# with κ = 2πk/ℓ. The products have no aliasing, so that the shift by s̃ along x̃,
# û_k → e^{iks̃} û_k, commutes with the truncated flow exactly. On x̃ the shift does not depend on
# ℓ, and ℓ is a parameter of the equation alone, with
#
#     ∂f/∂ℓ = -(1/ℓ) [(2κ² - 4κ⁴) û + N(û)],
#
# since dκ/dℓ = -κ/ℓ: the forced tangent, and the sensitivity, take ℓ as they take any parameter.
# The shift along x is s = s̃ ℓ/2π.
#
# A batch of m states is a 21 × m complex matrix, one state per column: seeds that the package
# takes as they are, with no code. The steps are those of CB3R2R3e of Flows.jl, of order three, Λ
# implicit. Each object of the dynamics holds its flow and its arrays, made once for the batches
# it advances.
using MultipleShooting
using FFTW
using LinearAlgebra
import Flows


# ---------------------------------------------------------------------------- #
# the equation                                                                 #
# ---------------------------------------------------------------------------- #

const KMAX  = 21 # the modes kept, k = 1, …, KMAX
const NGRID = 64 # the points of the grid of the products

# the wavenumbers κ = 2πk/ℓ of the modes kept on the domain of length ℓ, and the linear part
_wavenumbers(ℓ) = [2π*k/ℓ for k in 1:KMAX]
_linear(ℓ)      = _wavenumbers(ℓ).^2 .- _wavenumbers(ℓ).^4

# The transforms between the coefficients of b states and their values on the grid, with their
# arrays and plans, along the columns
struct Transforms{F, B}
      ŵ::Matrix{ComplexF64} # the coefficients of the wavenumbers 0, …, N/2, (N/2 + 1) × b
      w::Matrix{Float64}    # the values on the grid, N × b
    fwd::F                  # from the values to the coefficients
    bwd::B                  # from the coefficients to the values, not normalised

    function Transforms(b::Int)
        ŵ   = zeros(ComplexF64, NGRID ÷ 2 + 1, b)
        w   = zeros(NGRID, b)
        fwd = plan_rfft(w, 1)
        bwd = plan_brfft(ŵ, NGRID, 1)

        return new{typeof(fwd), typeof(bwd)}(ŵ,
                                             w,
                                             fwd,
                                             bwd)
    end
end

# the values on the grid of the states û, in T.w, with the mean and the modes above KMAX zero
function grid!(T::Transforms, û)
    fill!(T.ŵ, 0)
    view(T.ŵ, 2:(KMAX + 1), :) .= û
    return LinearAlgebra.mul!(T.w, T.bwd, T.ŵ)
end

# -c iκ times the coefficients of the modes kept of the values w, written into dû
function modes!(dû, T::Transforms, w, κ, c)
    LinearAlgebra.mul!(T.ŵ, T.fwd, w)
    dû .= (-c/NGRID) .* im .* κ .* view(T.ŵ, 2:(KMAX + 1), :)
    return dû
end

# The explicit term of the equation for m states, and of its tangent for k′ perturbations of every
# state, the k′ batches of m side by side along the columns, as Flows.jl calls them:
#
#     eq(t, û, dû)            dû = N(û) = -(iκ/2) (u²)^
#     eq(t, û, dû, v̂, dv̂)    dv̂ = N′(û) v̂ = -iκ (u v)^, every batch along the states, plus
#                             ∂f/∂ℓ for the forced tangent
struct Term{TU, TV}
         κ::Vector{Float64} # the wavenumbers of the modes kept
        Tu::TU              # the transforms of the states
        Tv::TV              # those of their perturbations, or nothing
         ℓ::Float64         # the length of the domain
    forced::Bool            # whether the tangent is forced by ∂f/∂ℓ
end

Term(ℓ, m; k′=0, forced=false) =
    Term(_wavenumbers(ℓ), Transforms(m), k′ > 0 ? Transforms(k′*m) : nothing, Float64(ℓ), forced)

function (eq::Term)(t, û, dû)
    u  = grid!(eq.Tu, û)
    u .= u.^2
    return modes!(dû, eq.Tu, u, eq.κ, 0.5)
end

function (eq::Term)(t, û, dû, v̂, dv̂)
    u = grid!(eq.Tu, û)
    v = grid!(eq.Tv, v̂)

    # ---- u v for every batch of perturbations: v seen as N × m × k′, each m columns times u ----
    reshape(v, NGRID, size(u, 2), :) .*= u
    modes!(dv̂, eq.Tv, v, eq.κ, 1.0)

    # ---- the forcing ∂f/∂ℓ, explicit, with dû = N(û) at the same stage ----
    eq.forced &&
        (dv̂ .-= ((2 .* eq.κ.^2 .- 4 .* eq.κ.^4) .* û .+ dû) ./ eq.ℓ)

    return dv̂
end

# The linear part Λ, implicit, as Flows.jl calls it
struct Linear
    Λ::Vector{Float64} # κ² - κ⁴, mode by mode
end

LinearAlgebra.mul!(out, A::Linear, X)  = (out .= A.Λ .* X; out)
Flows.ImcA!(A::Linear, c::Real, Y, Z) = (Z .= Y ./ (1 .- c .* A.Λ); Z)


# ---------------------------------------------------------------------------- #
# the objects of the dynamics                                                  #
# ---------------------------------------------------------------------------- #

# The flow G(X, τ): the m states of X advanced in place by τ
struct Flow{Φ}
     ℓ::Float64 # the length of the domain
     m::Int     # the states of a batch
    dt::Float64 # the time step
     ϕ::Φ       # the flow of Flows.jl, made once for batches of m states

    function Flow(ℓ::Real, m::Int; dt::Real=0.05)
        X = zeros(ComplexF64, KMAX, m)
        ϕ = Flows.flow(Term(ℓ, m),
                       Linear(_linear(ℓ)),
                       Flows.CB3R2R3e(X),
                       Flows.TimeStepConstant(dt))

        return new{typeof(ϕ)}(ℓ, m, dt, ϕ)
    end
end

(G::Flow)(X::Matrix{ComplexF64}, τ::Real) = (G.ϕ(X, (0, τ)); X)

# The tangent L(X, V, τ): the states advanced, and the perturbations V along them by the tangent
# equation, V a batch like X, for the Jacobian, or a vector of k′ of them, for the preconditioner;
# with `forced`, the forced tangent Lp(X, V, τ) of the parameter ℓ. The scheme is that of the
# coupled equations, so that the tangent is the derivative of the scheme, and all the directions
# go with the states in one integration
struct Tangent{Z, Φ}
         ℓ::Float64 # the length of the domain
         m::Int     # the states of a batch
        k′::Int     # the perturbations of every state
    forced::Bool    # whether the tangent equation is forced by ∂f/∂ℓ
        dt::Float64 # the time step
         Z::Z       # the states, 21 × m, coupled with their perturbations, 21 × k′m
         ϕ::Φ       # the flow of the coupled equations, made once

    function Tangent(ℓ::Real, m::Int; k′::Int=1, forced::Bool=false, dt::Real=0.05)
        Z  = Flows.couple(zeros(ComplexF64, KMAX, m), zeros(ComplexF64, KMAX, k′*m))
        eq = Term(ℓ, m; k′=k′, forced=forced)
        A  = Linear(_linear(ℓ))
        ϕ  = Flows.flow(Flows.couple(eq, eq),
                        Flows.couple(A, A),
                        Flows.CB3R2R3e(Z),
                        Flows.TimeStepConstant(dt))

        return new{typeof(Z), typeof(ϕ)}(ℓ, m, k′, forced, dt, Z, ϕ)
    end
end

function (P::Tangent)(X::Matrix{ComplexF64}, V::AbstractVector{<:Matrix{ComplexF64}}, τ::Real)
    # ---- input checks ----
    length(V) == P.k′ ||
        throw(ArgumentError("a tangent of $(P.k′) directions called with $(length(V))"))

    # ---- the states, and batch a of the perturbations in the columns (a - 1) m + 1, …, a m ----
    Y, W    = P.Z[1], P.Z[2]
    cols(a) = ((a - 1)*P.m + 1):(a*P.m)
    Y .= X
    for a in 1:P.k′
        view(W, :, cols(a)) .= V[a]
    end

    # ---- advanced together, the states once, and back ----
    P.ϕ(P.Z, (0, τ))
    X .= Y
    for a in 1:P.k′
        V[a] .= view(W, :, cols(a))
    end

    return V
end

(P::Tangent)(X::Matrix{ComplexF64}, V::Matrix{ComplexF64}, τ::Real) = (P(X, [V], τ); V)

# The right-hand side F(dX, X), Λ û + N(û) at the m states of X
struct RightHandSide{T}
     ℓ::Float64          # the length of the domain
     m::Int              # the states of a batch
     Λ::Vector{Float64}  # the linear part
    eq::T                # the explicit term, with its arrays for m states

    function RightHandSide(ℓ::Real, m::Int)
        eq = Term(ℓ, m)

        return new{typeof(eq)}(ℓ, m, _linear(ℓ), eq)
    end
end

(F::RightHandSide)(dX, X) = (F.eq(0.0, X, dX); dX .+= F.Λ .* X; dX)

# The shift S(X, s̃) along x̃, û_k e^{iks̃}, and its generator dS(dX, X, j), ∂/∂x̃, ik û_k, for the
# one shift: one object in two roles, with no arrays, the same on every domain
struct Shift
    k::Vector{Float64} # the wavenumbers along x̃ of the modes kept, 1, …, 21

    Shift() = new(collect(1.0:KMAX))
end

(S::Shift)(X, s::Real)    = (X .*= cis.(S.k .* s); X)
(S::Shift)(dX, X, j::Int) = (dX .= im .* S.k .* X; dX)

# All of them for m seeds on the domain of length ℓ: the tangent of the Jacobian, with one
# direction, that of the preconditioner, with k′, and the forced tangent
ks(ℓ; m=4, k′=4, dt=0.05) = (G  = Flow(ℓ, m; dt=dt),
                             L  = Tangent(ℓ, m; dt=dt),
                             Lk = Tangent(ℓ, m; k′=k′, dt=dt),
                             Lp = Tangent(ℓ, m; forced=true, dt=dt),
                             F  = RightHandSide(ℓ, m),
                             S  = Shift(),
                             dS = Shift())

# each of them on the domain of length ℓ, the family of the searches with ℓ as a parameter
at(G::Flow, ℓ)          = Flow(ℓ, G.m; dt=G.dt)
at(P::Tangent, ℓ)       = Tangent(ℓ, P.m; k′=P.k′, forced=P.forced, dt=P.dt)
at(F::RightHandSide, ℓ) = RightHandSide(ℓ, F.m)
at(S::Shift, ℓ)         = S

# the shift along x of the unknowns z on the domain of length ℓ
shift_x(z, ℓ) = shift(z, 1)*ℓ/2π

# ---- the guess of the tests at ℓ = 22: four seeds a quarter of the period apart along a
#      trajectory, at its closest return up to a shift, T = 16.8 and s = -2.92, the shift taken
#      along x̃ ----
include(joinpath(@__DIR__, "..", "..", "test", "data", "ks22.jl"))

const GUESS = ShootingVector(copy(ks22_guess.X), [period(ks22_guess), 2π*shift(ks22_guess, 1)/22])
