# GMRES, on its own: the solution of A x = b from x = 0, for a linear operator applied in place,
# A!(out, v) -> out, on any vector that broadcasts and has dot, norm and similar, such as a
# MultiVector. Each iteration adds one vector to the orthonormal basis Q of the Krylov space, by an
# Arnoldi step,
#
#     A Qₙ = Qₙ₊₁ Hₙ,   Hₙ upper Hessenberg, (n + 1) × n,
#
# and makes the new column of Hₙ triangular by a Givens rotation, which carries the right-hand
# side β e₁ into γ. The least-squares problem min ‖β e₁ - Hₙ y‖ is then the triangular system
# R y = γ[1:n], and its residual is |γₙ₊₁|, known without solving it: the iteration stops when it
# falls below tol β. No restarts: the basis grows up to maxiter vectors, one product with A each.
function _gmres(A!, b, tol::Real, maxiter::Int)
    β = LinearAlgebra.norm(b)

    # ---- b = 0 has the solution x = 0 ----
    β > 0 ||
        return (x = similar(b); x .= 0; x)

    # ---- the basis, the Hessenberg matrix, the rotations and the rotated right-hand side ----
    Q = [b ./ β]
    H = zeros(maxiter + 1, maxiter)
    c = zeros(maxiter)
    s = zeros(maxiter)
    γ = [β; zeros(maxiter)]
    n = 0

    # ---- one Arnoldi step and one rotation per iteration, until the residual is small ----
    while n < maxiter && abs(γ[n + 1]) > tol*β
        n += 1
        _arnoldi!(Q, H, A!, n)
        _givens!(H, γ, c, s, n)
    end

    abs(γ[n + 1]) > tol*β &&
        @warn "GMRES: relative residual $(abs(γ[n + 1])/β) after $n iterations"

    return _combination(Q, H, γ, n)
end

# The Arnoldi step n: the next vector of the basis from A qₙ, orthogonal to the basis by modified
# Gram–Schmidt, twice for orthogonality to rounding, and column n of H,
#
#     q = A qₙ - Σᵢ hᵢₙ qᵢ,   hᵢₙ = ⟨qᵢ, A qₙ⟩,   hₙ₊₁ₙ = ‖q‖,   qₙ₊₁ = q/hₙ₊₁ₙ
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
    q ./= H[n + 1, n]
    push!(Q, q)

    return Q
end

# Column n of H made triangular: the rotations of the columns before applied to it, then a new
# rotation that zeroes hₙ₊₁ₙ, applied to the right-hand side as well,
#
#     cₙ = hₙₙ/ρ,   sₙ = hₙ₊₁ₙ/ρ,   ρ = √(hₙₙ² + hₙ₊₁ₙ²)
function _givens!(H, γ, c, s, n)
    # ---- the rotations of the columns before ----
    for i in 1:n - 1
        H[i, n], H[i + 1, n] = _rotate(c[i], s[i], H[i, n], H[i + 1, n])
    end

    # ---- a new one, on the column and on the right-hand side ----
    ρ    = hypot(H[n, n], H[n + 1, n])
    c[n] = H[n, n]/ρ
    s[n] = H[n + 1, n]/ρ

    H[n, n], H[n + 1, n] = _rotate(c[n], s[n], H[n, n], H[n + 1, n])
    γ[n], γ[n + 1]       = _rotate(c[n], s[n], γ[n], γ[n + 1])

    return H
end

# the plane rotation of (x, y) by the cosine c and the sine s
#
#     [ c  s ] [ x ]
#     [-s  c ] [ y ]
_rotate(c, s, x, y) = (c*x + s*y, -s*x + c*y)

# the solution x = Σᵢ yᵢ qᵢ, with y from the triangular system R y = γ[1:n] left by the rotations
function _combination(Q, H, γ, n)
    y = LinearAlgebra.UpperTriangular(H[1:n, 1:n]) \ γ[1:n]

    x = similar(Q[1])
    x .= 0
    for i in 1:n
        x .+= y[i] .* Q[i]
    end

    return x
end
