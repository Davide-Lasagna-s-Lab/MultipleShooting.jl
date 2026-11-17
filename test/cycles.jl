# Two limit cycles that meet at a fold, for the tests of the parameter as an unknown: in polar
# coordinates r' = r (μ + r² - r⁴), θ' = 1, that is
#
#     x' = g x - y,   y' = g y + x,   g = μ + r² - r⁴,   r² = x² + y².
#
# For -1/4 < μ < 0 the circles r² = (1 ± √(1 + 4μ))/2 are orbits of period 2π. They meet at
# μ = -1/4, with r² = 1/2, and below there is none. Advanced by the RK4 of lorenz.jl, batched.

# ---- the right-hand side at μ, and its linearisation at x applied to v, column by column ----
function cycle_rhs!(dX::AbstractMatrix, X::AbstractMatrix, μ)
    for (dx, x) in zip(eachcol(dX), eachcol(X))
        r²    = x[1]^2 + x[2]^2
        g     = μ + r² - r²^2
        dx[1] = g*x[1] - x[2]
        dx[2] = g*x[2] + x[1]
    end

    return dX
end

function cycle_lin!(dV::AbstractMatrix, X::AbstractMatrix, V::AbstractMatrix, μ)
    for (dv, x, v) in zip(eachcol(dV), eachcol(X), eachcol(V))
        r²    = x[1]^2 + x[2]^2
        g     = μ + r² - r²^2
        dg    = 2*(1 - 2r²)*(x[1]*v[1] + x[2]*v[2])
        dv[1] = g*v[1] + dg*x[1] - v[2]
        dv[2] = g*v[2] + dg*x[2] + v[1]
    end

    return dV
end

struct CyclePropagator
    dt::Float64 # the largest time step
     μ::Float64 # the parameter μ
end

MultipleShooting.propagate!(P::CyclePropagator, X::Batched, τ::Real) =
    (rk4!((dX, Y) -> cycle_rhs!(dX, Y, P.μ), X, τ, P.dt); X)

MultipleShooting.tangent!(P::CyclePropagator, X::Batched, V::Batched, τ::Real) =
    (rk4!((dX, Y) -> cycle_rhs!(dX, Y, P.μ), (dV, Y, W) -> cycle_lin!(dV, Y, W, P.μ), X, V, τ, P.dt); V)

# the tangent equation forced by the derivative of the right-hand side in μ, ∂f/∂μ = (x, y)
MultipleShooting.forced!(P::CyclePropagator, X::Batched, V::Batched, τ::Real) =
    (rk4!((dX, Y) -> cycle_rhs!(dX, Y, P.μ), (dV, Y, W) -> (cycle_lin!(dV, Y, W, P.μ); dV .+= Y; dV), X, V, τ, P.dt); V)

MultipleShooting.rhs!(P::CyclePropagator, F::Batched, X::Batched) = cycle_rhs!(F, X, P.μ)

# the circle of squared radius r² as four seeds a quarter of a turn apart, with the period 2π
circle(r²) = MultiVector(sqrt(r²) .* [1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [2π])
