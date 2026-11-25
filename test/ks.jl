# The Kuramoto–Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx on [0, L) with periodic boundary
# conditions, in the full space with its continuous translation symmetry, advanced by Flows.jl.
#
# A state is the Fourier coefficients û_k of the modes kept, k = 1, …, K, with K the last below
# N/3 for products on a grid of N points,
#
#     u(x) = Σ_{0 < |k| ≤ K} û_k e^{iκx},   κ = 2πk/L,   û_{-k} = conj(û_k),
#
# and a batch is a K × m complex matrix with one state per column. The mean, which the flow
# conserves, is zero, and the modes that the quadratic term would alias are left out, so that the
# truncated system keeps the translation symmetry exactly,
#
#     S(s) u(x) = u(x + s),   û_k → e^{iκ s} û_k,   with the generator ∂ₓ,   û_k → iκ û_k.
#
# The equation is û_t = Λ û + N(û), with Λ = κ² - κ⁴ and N(û) = -(iκ/2) (u²)^. The steps are those
# of CB3R2R3e of Flows.jl, of order three, with Λ implicit, equal and of at most dt. The tangent
# is advanced with the states as one coupled state, by the linearisation of the explicit term
# along the stages of the states, so that it is the exact derivative of the discrete map.
using FFTW
import Flows

struct KSPropagator
     L::Float64            # the length of the domain
    dt::Float64            # the largest time step
     N::Int                # the points of the grid of the products
     κ::Vector{Float64}    # the wavenumbers of the modes kept, 2πk/L for 1 ≤ k < N/3

    KSPropagator(; L::Real=22.0, N::Int=64, dt::Real=0.05) =
        new(L,
            dt,
            N,
            [2π*k/L for k in 1:N÷2 if k < N/3])
end

# the linear part of the equation, Λ = κ² - κ⁴ on every mode, as the implicit term of Flows.jl
struct KSLinear
    Λ::Vector{Float64}     # κ² - κ⁴, mode by mode
end

KSLinear(P::KSPropagator) = KSLinear(P.κ.^2 .- P.κ.^4)

LinearAlgebra.mul!(out, A::KSLinear, X)         = (out .= A.Λ .* X; out)
Flows.ImcA!(A::KSLinear, c::Real, Y, Z)         = (Z .= Y ./ (1 .- c .* A.Λ); Z)

# The explicit terms of the equation and of its tangent, for batches like X, as a flow of Flows.jl
# calls them, with the products on the grid of N points, by transforms along the first dimension:
#
#     nonlinear!(t, û, dû)            dû = -(iκ/2) (u²)^
#     linearised!(t, û, dû, v̂, dv̂)    dv̂ = -iκ (u v)^
function ks_terms(P::KSPropagator, X::AbstractMatrix)
    K, m = size(X)

    # ---- the half spectrum of the grid, two fields on it, and the transforms ----
    ŵ   = zeros(ComplexF64, P.N÷2 + 1, m)
    u   = zeros(P.N, m)
    v   = zeros(P.N, m)
    fwd = plan_rfft(u, 1)
    bwd = plan_brfft(ŵ, P.N, 1)

    # the field of the coefficients û on the grid, into w: the transform destroys its input
    function grid!(w, û)
        fill!(ŵ, 0)
        view(ŵ, 2:K+1, :) .= û

        return mul!(w, bwd, ŵ)
    end

    # -c iκ times the coefficients of the field w on the grid, on the modes kept, into dû
    function modes!(dû, w, c)
        mul!(ŵ, fwd, w)
        dû .= (-c/P.N) .* im .* P.κ .* view(ŵ, 2:K+1, :)

        return dû
    end

    nonlinear!(t, û, dû)         = (grid!(u, û); u .= u.^2; modes!(dû, u, 0.5))
    linearised!(t, û, dû, v̂, dv̂) = (grid!(u, û); grid!(v, v̂); u .*= v; modes!(dv̂, u, 1.0))

    return nonlinear!, linearised!
end

# the equal steps over τ, of at most dt
ks_steps(P::KSPropagator, τ) = Flows.TimeStepConstant(τ/max(1, ceil(Int, τ/P.dt)))

function MultipleShooting.propagate!(P::KSPropagator, X::AbstractMatrix, τ::Real)
    # ---- the flow works on a copy, of the type of its registers: X may be a view ----
    Y    = copy(X)
    N!,  = ks_terms(P, Y)
    ϕ    = Flows.flow(N!, KSLinear(P), Flows.CB3R2R3e(Y), ks_steps(P, τ))
    ϕ(Y, (0.0, τ))
    X   .= Y

    return X
end

function MultipleShooting.tangent!(P::KSPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real)
    # ---- the states and the vectors as one coupled state, on copies ----
    Z      = Flows.couple(copy(X), copy(V))
    N!, L! = ks_terms(P, Z[1])
    A      = KSLinear(P)
    ϕ      = Flows.flow(Flows.couple(N!, L!), Flows.couple(A, A), Flows.CB3R2R3e(Z), ks_steps(P, τ))
    ϕ(Z, (0.0, τ))
    X     .= Z[1]
    V     .= Z[2]

    return V
end

MultipleShooting.tangent!(P::KSPropagator, X::AbstractMatrix, V::AbstractVector{<:AbstractMatrix}, τ::Real) =
    directions!(P, X, V, τ)

function MultipleShooting.rhs!(P::KSPropagator, F::AbstractMatrix, X::AbstractMatrix)
    # ---- Λ û + N(û) ----
    N!, = ks_terms(P, X)
    N!(0.0, X, F)
    F .+= (P.κ.^2 .- P.κ.^4) .* X

    return F
end

# û_k e^{iκ s}, the one shift s[1]
MultipleShooting.shift!(P::KSPropagator, X::AbstractMatrix, s) = (X .*= cis.(P.κ .* s[1]); X)

# the generator ∂ₓ, iκ û_k
MultipleShooting.dshift!(P::KSPropagator, G::AbstractMatrix, X::AbstractMatrix, j::Int) =
    (G .= im .* P.κ .* X; G)
