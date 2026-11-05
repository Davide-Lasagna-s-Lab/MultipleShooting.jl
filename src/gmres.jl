# GMRES, on its own: the solution of A x = b from x = 0, for a linear operator applied in place,
# A!(out, v) -> out, on any vector that broadcasts and has dot, norm and similar, such as a
# MultiVector. Each iteration adds one vector to the orthonormal basis Q of the Krylov space, by an
# Arnoldi step,
#
#     A Qₙ = Qₙ₊₁ Hₙ,   Hₙ upper Hessenberg, (n + 1) × n,
#
# and makes the new column of a copy R of Hₙ triangular by a Givens rotation, which carries the
# right-hand side β e₁ into γ. The least-squares problem min ‖β e₁ - Hₙ y‖ is then the triangular
# system R y = γ[1:n], and its residual is |γₙ₊₁|, known without solving it: the iteration stops
# when it falls below tol β. No restarts: the basis grows up to maxiter vectors, one product with A
# each. Hₙ itself is kept, so that the run leaves its history: the Ritz values of every iteration
# from the leading blocks of Hₙ, and the residual of every iteration.

"""
    gmres(A!, b; tol=1e-8, maxiter=100) -> (x, H, residuals)

The solution of `A x = b` by GMRES from `x = 0`, without restarts, with the history of the run.
It stops when the residual falls below `tol ‖b‖`, or after `maxiter` iterations, one product
with `A` each. The Hessenberg matrix `H` of the Arnoldi relation `A Qₙ = Qₙ₊₁ H` holds in its
leading square blocks the Ritz values of every iteration, by [`ritz_values`](@ref).

# Arguments
- `A!`: the linear operator, applied in place as `A!(out, v) -> out`.
- `b`: the right-hand side, not changed, any vector that broadcasts and has `dot`, `norm` and
  `similar`, such as a [`MultiVector`](@ref).
- `tol::Real=1e-8`: the relative residual at which it stops.
- `maxiter::Int=100`: the most iterations.

# Returns
A named tuple with
- `x`: the solution after the last iteration `n`.
- `H::Matrix{Float64}`: the upper Hessenberg matrix, `(n + 1) × n`.
- `residuals::Vector{Float64}`: the relative residuals `‖b - A xⱼ‖/‖b‖`, `j = 0, …, n`.
"""
function gmres(A!, b; tol::Real=1e-8, maxiter::Int=100)
    β = LinearAlgebra.norm(b)

    # ---- b = 0 has the solution x = 0, with no iterations ----
    if β == 0
        x  = similar(b)
        x .= 0

        return (x         = x,
                H         = zeros(1, 0),
                residuals = [0.0])
    end

    # ---- the basis, the Hessenberg matrix and its rotated copy, the rotations, the rotated
    #      right-hand side and the relative residuals ----
    Q = [b ./ β]
    H = zeros(maxiter + 1, maxiter)
    R = zeros(maxiter + 1, maxiter)
    c = zeros(maxiter)
    s = zeros(maxiter)
    γ = [β; zeros(maxiter)]
    r = [1.0]
    n = 0

    # ---- one Arnoldi step and one rotation per iteration, until the residual is small ----
    while n < maxiter && r[end] > tol
        n += 1
        _arnoldi!(Q, H, A!, n)

        R[1:(n + 1), n] .= H[1:(n + 1), n]
        _givens!(R, γ, c, s, n)

        push!(r, abs(γ[n + 1])/β)
    end

    r[end] > tol &&
        @warn "GMRES: relative residual $(r[end]) after $n iterations"

    return (x         = _combination(Q, R, γ, n),
            H         = H[1:(n + 1), 1:n],
            residuals = r)
end

"""
    ritz_values(H, n=size(H, 2))

The Ritz values of the operator after `n` iterations of [`gmres`](@ref): the eigenvalues of the
leading `n × n` block of its Hessenberg matrix, `Qₙᵀ A Qₙ`, the operator restricted to the
Krylov space of dimension `n`. Complex in general, sorted by real part, then imaginary part.

# Arguments
- `H::AbstractMatrix`: the Hessenberg matrix that [`gmres`](@ref) returns.
- `n::Int=size(H, 2)`: the iteration, at most the number of columns of `H`.
"""
ritz_values(H::AbstractMatrix, n::Int=size(H, 2)) = LinearAlgebra.eigvals(H[1:n, 1:n])

# The Arnoldi step n: the next vector of the basis from A qₙ, orthogonal to the basis by modified
# Gram–Schmidt, twice for orthogonality to rounding, and column n of H,
#
#     q = A qₙ - Σᵢ hᵢₙ qᵢ,   hᵢₙ = ⟨qᵢ, A qₙ⟩,   hₙ₊₁ₙ = ‖q‖,   qₙ₊₁ = q/hₙ₊₁ₙ
#
# A zero q means that the Krylov space is invariant, and that the residual is now zero: q is then
# kept as it is, and never used.
function _arnoldi!(Q, H, A!, n)
    q = A!(similar(Q[n]), Q[n])

    # ---- minus its components along the basis ----
    for _ in 1:2, i in 1:n
        h        = LinearAlgebra.dot(Q[i], q)
        q      .-= h .* Q[i]
        H[i, n] += h
    end

    # ---- normalised, and added to the basis ----
    H[n + 1, n] = LinearAlgebra.norm(q)
    H[n + 1, n] > 0 && (q ./= H[n + 1, n])
    push!(Q, q)

    return Q
end

# Column n of R made triangular: the rotations of the columns before applied to it, then a new
# rotation that zeroes rₙ₊₁ₙ, applied to the right-hand side as well,
#
#     cₙ = rₙₙ/ρ,   sₙ = rₙ₊₁ₙ/ρ,   ρ = √(rₙₙ² + rₙ₊₁ₙ²)
function _givens!(R, γ, c, s, n)
    # ---- the rotations of the columns before ----
    for i in 1:n - 1
        R[i, n], R[i + 1, n] = _rotate(c[i], s[i], R[i, n], R[i + 1, n])
    end

    # ---- a new one, on the column and on the right-hand side ----
    ρ    = hypot(R[n, n], R[n + 1, n])
    c[n] = R[n, n]/ρ
    s[n] = R[n + 1, n]/ρ

    R[n, n], R[n + 1, n] = _rotate(c[n], s[n], R[n, n], R[n + 1, n])
    γ[n], γ[n + 1]       = _rotate(c[n], s[n], γ[n], γ[n + 1])

    return R
end

# the plane rotation of (x, y) by the cosine c and the sine s
#
#     [ c  s ] [ x ]
#     [-s  c ] [ y ]
_rotate(c, s, x, y) = (c*x + s*y, -s*x + c*y)

# the solution x = Σᵢ yᵢ qᵢ, with y from the triangular system R y = γ[1:n] left by the rotations
function _combination(Q, R, γ, n)
    y = LinearAlgebra.UpperTriangular(R[1:n, 1:n]) \ γ[1:n]

    x = similar(Q[1])
    x .= 0
    for i in 1:n
        x .+= y[i] .* Q[i]
    end

    return x
end
