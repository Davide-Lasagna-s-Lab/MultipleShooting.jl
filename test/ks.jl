# The Kuramoto–Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx on [0, L) with periodic boundary
# conditions, in the full space with its continuous translation symmetry. A state is u on N grid
# points, x_j = j L/N, and a batch is an N × m matrix with one state per column; all the work is
# done on the whole batch at once, with the transforms along the first dimension.
#
# In Fourier space the equation is û_t = Λ û + N(û), with Λ = κ² - κ⁴ and N(û) = -(iκ/2) (u²)^.
# Only the modes 1 ≤ k < N/3 are kept: the mean, which the flow conserves, and the modes the
# quadratic term would alias are set to zero, so that the truncated system keeps the translation
# symmetry exactly,
#
#     S(s) u(x) = u(x + s),   û_k → e^{iκ s} û_k,   with the generator ∂ₓ,   û_k → iκ û_k.
#
# The steps are ETDRK4, the exponential Runge–Kutta scheme of order four, with its coefficients
# from a mean over points on a circle around hΛ, which avoids their cancellation at small hΛ. The
# tangent is the linearisation of the scheme, stage by stage along the stages of the states, so
# that it is the exact derivative of the discrete map.
using FFTW

struct KSPropagator
       L::Float64              # the length of the domain
      dt::Float64              # the largest time step
       Λ::Vector{Float64}      # the linear part κ² - κ⁴, mode by mode, k = 0, …, N/2
    keep::Vector{Float64}      # 1 on the modes kept, 1 ≤ k < N/3, and 0 elsewhere
       D::Vector{ComplexF64}   # iκ on the modes kept, ∂ₓ, and 0 elsewhere

    function KSPropagator(; L::Real=22.0,
                            N::Int=64,
                            dt::Real=0.05)
        # ---- the wavenumbers of the real transform, and the modes kept ----
        k    = 0:N÷2
        κ    = 2π .* k ./ L
        keep = Float64.(1 .<= k .< N/3)

        return new(L,
                   dt,
                   κ.^2 .- κ.^4,
                   keep,
                   im .* κ .* keep)
    end
end

# the transforms along the first dimension for a batch of the size of X, and the buffers they work
# on: two on the grid, and one in Fourier space, which the inverse transform overwrites
function _ks_work(X::AbstractMatrix)
    N, m = size(X)
    u    = Matrix{Float64}(undef, N, m)
    ŵ    = Matrix{ComplexF64}(undef, N÷2 + 1, m)

    return (f = plan_rfft(u, 1),
            b = plan_irfft(ŵ, N, 1),
            u = u,
            w = similar(u),
            ŵ = ŵ)
end

# from the grid to the modes kept, and back
function _ks_fourier!(v̂, P::KSPropagator, W, X)
    W.u .= X
    mul!(v̂, W.f, W.u)
    v̂ .*= P.keep

    return v̂
end

function _ks_grid!(X, W, v̂)
    W.ŵ .= v̂
    mul!(W.u, W.b, W.ŵ)
    X .= W.u

    return X
end

# the nonlinear term N(v̂) = -(u²/2)ₓ on the modes kept
function _ks_nonlinear!(N̂, P::KSPropagator, W, v̂)
    _ks_grid!(W.u, W, v̂)
    W.u .= W.u.^2 ./ 2
    mul!(N̂, W.f, W.u)
    N̂ .*= .-P.D

    return N̂
end

# and its linearisation at v̂ along δv̂, -(u δu)ₓ. δu first, since _ks_grid! passes through W.u
function _ks_nonlinear!(N̂, P::KSPropagator, W, v̂, δv̂)
    _ks_grid!(W.w, W, δv̂)
    _ks_grid!(W.u, W, v̂)
    W.u .= W.u .* W.w
    mul!(N̂, W.f, W.u)
    N̂ .*= .-P.D

    return N̂
end

# the ETDRK4 coefficients of the step h, mode by mode: each function of hΛ is the mean of the
# same function over M points on the upper half of a circle of radius one around hΛ, real part
function _etdrk4(Λ, h; M::Int=32)
    z = h .* Λ .+ transpose(cispi.(((1:M) .- 0.5) ./ M))

    # ---- the mean over the circle, real part ----
    mean(F) = vec(real(sum(F; dims=2))) ./ M

    E  = exp.(h .* Λ)
    E₂ = exp.(h .* Λ ./ 2)
    Q  = h .* mean((exp.(z ./ 2) .- 1) ./ z)
    f₁ = h .* mean((-4 .- z .+ exp.(z) .* (4 .- 3 .* z .+ z.^2)) ./ z.^3)
    f₂ = h .* mean((2 .+ z .+ exp.(z) .* (z .- 2)) ./ z.^3)
    f₃ = h .* mean((-4 .- 3 .* z .- z.^2 .+ exp.(z) .* (4 .- z)) ./ z.^3)

    return E, E₂, Q, f₁, f₂, f₃
end

function MultipleShooting.propagate!(P::KSPropagator, X::AbstractMatrix, τ::Real)
    n = max(1, ceil(Int, τ/P.dt))
    E, E₂, Q, f₁, f₂, f₃ = _etdrk4(P.Λ, τ/n)

    # ---- the transforms, the states in Fourier space, and their stages ----
    W = _ks_work(X)
    v̂ = _ks_fourier!(similar(W.ŵ), P, W, X)
    â, b̂, ĉ        = similar(v̂), similar(v̂), similar(v̂)
    Nv, Na, Nb, Nc = similar(v̂), similar(v̂), similar(v̂), similar(v̂)

    for _ in 1:n
        _ks_nonlinear!(Nv, P, W, v̂)
        â .= E₂ .* v̂ .+ Q .* Nv
        _ks_nonlinear!(Na, P, W, â)
        b̂ .= E₂ .* v̂ .+ Q .* Na
        _ks_nonlinear!(Nb, P, W, b̂)
        ĉ .= E₂ .* â .+ Q .* (2 .* Nb .- Nv)
        _ks_nonlinear!(Nc, P, W, ĉ)
        v̂ .= E .* v̂ .+ f₁ .* Nv .+ 2 .* f₂ .* (Na .+ Nb) .+ f₃ .* Nc
    end

    return _ks_grid!(X, W, v̂)
end

function MultipleShooting.tangent!(P::KSPropagator, X::AbstractMatrix, V::AbstractMatrix, τ::Real)
    n = max(1, ceil(Int, τ/P.dt))
    E, E₂, Q, f₁, f₂, f₃ = _etdrk4(P.Λ, τ/n)

    # ---- the states once beside each of the k′ blocks of V, since a stage pairs a state with a
    #      vector ----
    Y = repeat(X, 1, size(V, 2) ÷ size(X, 2))

    # ---- the transforms, the states and the vectors in Fourier space, and their stages ----
    W  = _ks_work(Y)
    v̂  = _ks_fourier!(similar(W.ŵ), P, W, Y)
    δv̂ = _ks_fourier!(similar(W.ŵ), P, W, V)
    â, b̂, ĉ        = similar(v̂), similar(v̂), similar(v̂)
    δâ, δb̂, δĉ     = similar(v̂), similar(v̂), similar(v̂)
    Nv, Na, Nb, Nc = similar(v̂), similar(v̂), similar(v̂), similar(v̂)
    Lv, La, Lb, Lc = similar(v̂), similar(v̂), similar(v̂), similar(v̂)

    for _ in 1:n
        # ---- the stages of the states ----
        _ks_nonlinear!(Nv, P, W, v̂)
        â .= E₂ .* v̂ .+ Q .* Nv
        _ks_nonlinear!(Na, P, W, â)
        b̂ .= E₂ .* v̂ .+ Q .* Na
        _ks_nonlinear!(Nb, P, W, b̂)
        ĉ .= E₂ .* â .+ Q .* (2 .* Nb .- Nv)
        _ks_nonlinear!(Nc, P, W, ĉ)

        # ---- the same stages linearised, along those of the states ----
        _ks_nonlinear!(Lv, P, W, v̂, δv̂)
        δâ .= E₂ .* δv̂ .+ Q .* Lv
        _ks_nonlinear!(La, P, W, â, δâ)
        δb̂ .= E₂ .* δv̂ .+ Q .* La
        _ks_nonlinear!(Lb, P, W, b̂, δb̂)
        δĉ .= E₂ .* δâ .+ Q .* (2 .* Lb .- Lv)
        _ks_nonlinear!(Lc, P, W, ĉ, δĉ)

        δv̂ .= E .* δv̂ .+ f₁ .* Lv .+ 2 .* f₂ .* (La .+ Lb) .+ f₃ .* Lc
        v̂  .= E .* v̂ .+ f₁ .* Nv .+ 2 .* f₂ .* (Na .+ Nb) .+ f₃ .* Nc
    end

    # ---- back on the grid, the states from the first block ----
    _ks_grid!(Y, W, v̂)
    X .= view(Y, :, axes(X, 2))

    return _ks_grid!(V, W, δv̂)
end

function MultipleShooting.rhs!(P::KSPropagator, F::AbstractMatrix, X::AbstractMatrix)
    # ---- Λ û + N(û), on the modes kept ----
    W = _ks_work(X)
    v̂ = _ks_fourier!(similar(W.ŵ), P, W, X)
    N̂ = _ks_nonlinear!(similar(v̂), P, W, v̂)

    return _ks_grid!(F, W, P.Λ .* v̂ .+ N̂)
end

function MultipleShooting.shift!(P::KSPropagator, X::AbstractMatrix, s)
    # ---- û_k e^{iκ s}, the one shift s[1] ----
    W = _ks_work(X)
    v̂ = _ks_fourier!(similar(W.ŵ), P, W, X)

    return _ks_grid!(X, W, exp.(P.D .* s[1]) .* v̂)
end

function MultipleShooting.dshift!(P::KSPropagator, G::AbstractMatrix, X::AbstractMatrix, j::Int)
    # ---- the generator ∂ₓ, iκ û_k ----
    W = _ks_work(X)
    v̂ = _ks_fourier!(similar(W.ŵ), P, W, X)

    return _ks_grid!(G, W, P.D .* v̂)
end
