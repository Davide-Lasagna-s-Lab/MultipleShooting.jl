# Two limit cycles that meet at a fold, for the tests of the parameter as an unknown: in polar
# coordinates r' = r (μ + r² - r⁴), θ' = 1, that is
#
#     x' = g x - y,   y' = g y + x,   g = μ + r² - r⁴,   r² = x² + y².
#
# For -1/4 < μ < 0 the circles r² = (1 ± √(1 + 4μ))/2 are orbits of period 2π. They meet at
# μ = -1/4, with r² = 1/2, and below there is none. Advanced by the RK4 of Flows.jl, as in lorenz.jl.

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

# The objects of the dynamics at μ, each a Cycle{R} of its role R, as Lorenz{R} of lorenz.jl: the
# flow :G, the tangent :L, the right-hand side :F, and the tangent forced by ∂f/∂μ = (x, y) :Lp
struct Cycle{R}
    dt::Float64 # the largest time step
     μ::Float64 # the parameter μ
end

# all of them at μ, in a named tuple
cycles(μ; dt=1e-2) = (G  = Cycle{:G}(dt, μ),
                      L  = Cycle{:L}(dt, μ),
                      F  = Cycle{:F}(dt, μ),
                      Lp = Cycle{:Lp}(dt, μ))

# one of them at another μ, the family of the searches with a parameter
at(x::Cycle{R}, μ) where {R} = Cycle{R}(x.dt, μ)

# the right-hand side and its linearisation at the μ of x, as a flow of Flows.jl calls them
_rhs(x::Cycle) = (t, X, dX) -> cycle_rhs!(dX, X, x.μ)
_lin(x::Cycle) = (t, X, dX, V, dV) -> cycle_lin!(dV, X, V, x.μ)

(G::Cycle{:G})(X::Batched, τ::Real) = advance!(_rhs(G), X, τ, G.dt)

(L::Cycle{:L})(X::Batched, V::Batched, τ::Real) = advance!(_rhs(L), _lin(L), X, V, τ, L.dt)

(L::Cycle{:L})(X::Batched, V::AbstractVector{<:Batched}, τ::Real) = directions!(L, X, V, τ)

(Lp::Cycle{:Lp})(X::Batched, V::Batched, τ::Real) =
    advance!(_rhs(Lp), (t, Y, dY, W, dW) -> (cycle_lin!(dW, Y, W, Lp.μ); dW .+= Y; dW), X, V, τ, Lp.dt)

(F::Cycle{:F})(dX::Batched, X::Batched) = cycle_rhs!(dX, X, F.μ)

# the circle of squared radius r² as four seeds a quarter of a turn apart, with the period 2π
circle(r²) = ShootingVector(sqrt(r²) .* [1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [2π])
