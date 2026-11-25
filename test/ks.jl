# The Kuramoto–Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx on [0, ℓ) with periodic boundary
# conditions, in the full space with its continuous translation symmetry, advanced by Flows.jl.
#
# A state is the Fourier coefficients û_k of the modes kept, k = 1, …, K, with K the last below
# N/3 for products on a grid of N points,
#
#     u(x) = Σ_{0 < |k| ≤ K} û_k e^{iκx},   κ = 2πk/ℓ,   û_{-k} = conj(û_k),
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

# The objects of the dynamics on the domain of length ℓ, each a KS{R} of its role R, as Lorenz{R}
# of lorenz.jl: the flow :G, the tangent :L, the right-hand side :F, the shift :S and its
# generator :dS
struct KS{R}
     ℓ::Float64            # the length of the domain
    dt::Float64            # the largest time step
     N::Int                # the points of the grid of the products
     κ::Vector{Float64}    # the wavenumbers of the modes kept, 2πk/ℓ for 1 ≤ k < N/3

    KS{R}(; ℓ::Real=22.0, N::Int=64, dt::Real=0.05) where {R} =
        new{R}(ℓ,
               dt,
               N,
               [2π*k/ℓ for k in 1:N÷2 if k < N/3])
end

# all of them, in a named tuple
ks(; kw...) = (G  = KS{:G}(; kw...),
               L  = KS{:L}(; kw...),
               F  = KS{:F}(; kw...),
               S  = KS{:S}(; kw...),
               dS = KS{:dS}(; kw...))

# one of them on a domain of another length, the family of the searches with a parameter
at(x::KS{R}, ℓ) where {R} = KS{R}(; ℓ=ℓ, N=x.N, dt=x.dt)

# the linear part of the equation, Λ = κ² - κ⁴ on every mode, as the implicit term of Flows.jl
struct KSLinear
    Λ::Vector{Float64}     # κ² - κ⁴, mode by mode
end

KSLinear(P::KS) = KSLinear(P.κ.^2 .- P.κ.^4)

LinearAlgebra.mul!(out, A::KSLinear, X)         = (out .= A.Λ .* X; out)
Flows.ImcA!(A::KSLinear, c::Real, Y, Z)         = (Z .= Y ./ (1 .- c .* A.Λ); Z)

# The explicit terms of the equation and of its tangent, for batches like X, as a flow of Flows.jl
# calls them, with the products on the grid of N points, by transforms along the first dimension:
#
#     nonlinear!(t, û, dû)            dû = -(iκ/2) (u²)^
#     linearised!(t, û, dû, v̂, dv̂)    dv̂ = -iκ (u v)^
function ks_terms(P::KS, X::AbstractMatrix)
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
ks_steps(P::KS, τ) = Flows.TimeStepConstant(τ/max(1, ceil(Int, τ/P.dt)))

function (P::KS{:G})(X::Batched, τ::Real)
    # ---- the flow works on a copy, of the type of its registers: X may be a view ----
    Y    = copy(X)
    N!,  = ks_terms(P, Y)
    ϕ    = Flows.flow(N!, KSLinear(P), Flows.CB3R2R3e(Y), ks_steps(P, τ))
    ϕ(Y, (0.0, τ))
    X   .= Y

    return X
end

function (P::KS{:L})(X::Batched, V::Batched, τ::Real)
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

(P::KS{:L})(X::Batched, V::AbstractVector{<:Batched}, τ::Real) = directions!(P, X, V, τ)

function (P::KS{:F})(F::Batched, X::Batched)
    # ---- Λ û + N(û) ----
    N!, = ks_terms(P, X)
    N!(0.0, X, F)
    F .+= (P.κ.^2 .- P.κ.^4) .* X

    return F
end

# û_k e^{iκ s}, the one shift s
(P::KS{:S})(X::Batched, s::Real) = (X .*= cis.(P.κ .* s); X)

# the generator ∂ₓ, iκ û_k
(P::KS{:dS})(G::Batched, X::Batched, j::Int) = (G .= im .* P.κ .* X; G)
