# RK4 with a fixed number of steps for the small systems of the tests: one state x, and the tangent
# vector v if not nothing, advanced over τ in steps of at most dt, the tangent by the
# linearisation of the scheme. rhs!(dx, x) is the right-hand side, jacobian(x) its Jacobian matrix.
function rk4!(rhs!, jacobian, x, v, τ, dt)
    n = max(1, ceil(Int, τ/dt))
    h = τ/n
    f(x) = rhs!(similar(x), x)
    for _ in 1:n
        x2 = x .+ (h/2) .* f(x)
        x3 = x .+ (h/2) .* f(x2)
        x4 = x .+ h .* f(x3)
        if v !== nothing
            l1 = jacobian(x)*v
            l2 = jacobian(x2)*(v .+ (h/2) .* l1)
            l3 = jacobian(x3)*(v .+ (h/2) .* l2)
            l4 = jacobian(x4)*(v .+ h .* l3)
            v .+= (h/6) .* (l1 .+ 2 .* l2 .+ 2 .* l3 .+ l4)
        end
        x .+= (h/6) .* (f(x) .+ 2 .* f(x2) .+ 2 .* f(x3) .+ f(x4))
    end

    return x
end
