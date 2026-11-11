# A right preconditioner of the Jacobian for Newton–Krylov, on the leading directions of the loop.
#
# With the reordered rows of residual.jl the Jacobian is A = -I + C + B: -I on the seeds, C the
# cyclic block subdiagonal of the tangent maps, ψᵢ from seed i to seed i + 1 and S(s) ψ_m from the
# last seed to the first, and B the columns of the period and of the shifts with the rows of the
# phase conditions. The directions that the loop of tangent maps does not contract make GMRES slow,
# about m iterations each. The bases of a subspace iteration hold them: with Uᵢ the orthonormal
# basis of k directions at seed i,
#
#     W = blockdiag(U₁, …, U_m, I),   E = Wᵀ A W,   M⁻¹ = -(I - W Wᵀ) + W E⁻¹ Wᵀ,
#
# the identity on the period and the shifts. M⁻¹ is -I, the inverse of the diagonal of A, outside
# the span of W, and an exact solve with E on it.
#
# The coarse matrix E comes with no integration of its own. After a subspace step the bases before
# it, Uᵢ, and the coordinates of their images in them, (UᵀV)ᵢ₊₁ = Uᵢ₊₁ᵀ ψᵢ Uᵢ, are both at hand, so
# that W is built from U and
#
#     E[i + 1, i] = Uᵢ₊₁ᵀ ψᵢ Uᵢ,   E[1, m] = U₁ᵀ S(s) ψ_m U_m,   E[i, i] = -I,
#     E[i + 1, T] = wᵢ Uᵢ₊₁ᵀ f(φ_τ(uᵢ)),   E[1, sⱼ] = U₁ᵀ gⱼ(S(s) φ_τ(u_m)),
#     E[T, 1] = f(u₁)ᵀ U₁,   E[sⱼ, 1] = gⱼ(u₁)ᵀ U₁,
#
# with f and the generators at the end points of the segments and at the first seed: the column of
# the period is f at the end points, the default of dpropagate!, enough for a preconditioner. The
# subspace step integrates the seeds along with the bases, and leaves the end points φ_τ(uᵢ), so
# that these need no integration of their own: an update of the preconditioner integrates the
# seeds k times, those of its subspace step. E has m k + 1 + nshifts rows, and is factorised by
# sparse QR, since Gaussian elimination is unstable on cyclic block-bidiagonal matrices.

"""
    Preconditioner(P, z; k, tol=1e-8, maxsteps=100, p=k)

A right preconditioner of the Jacobian of the propagator `P` at `z`, on the leading directions
of the loop, from a [`Subspace`](@ref) along it, with random bases at the start: with Uᵢ the
bases of `k` directions at the seeds,

    W = blockdiag(U₁, …, U_m, I),   E = Wᵀ J W,   M⁻¹ = -(I - W Wᵀ) + W E⁻¹ Wᵀ,

-I outside the span of W and an exact solve with E on it. Every time it
is built or moved, its subspace makes steps until the bases are accurate. For
`nkroot!(F, J, z; precond=M)` of NKRoots.jl: `update!(M, z)` moves it to `z`, `ldiv!(out, M, v)`
applies `M⁻¹`, and `gram(M, Q)` gives the root of the Gram matrix of the vectors `M⁻¹ qⱼ`. Every
subspace step integrates the seeds k times.

# Arguments
- `P`: the propagator, with `tangent!` and `rhs!`, and `shift!` and `dshift!` for relative
  periodic orbits.
- `z::MultiVector`: the point, kept, not copied.
- `k::Int`: the number of directions per seed.
- `tol::Real=1e-8`: the accuracy of the bases, absolute, as [`change`](@ref).
- `maxsteps::Int=100`: the most subspace steps per update, an error if the bases are not accurate
  by then; `M.S.steps` holds the number of the last one.
- `p::Int=k`: the number of leading directions whose accuracy is tested.
"""
mutable struct Preconditioner{ST<:Subspace}
           S::ST                                         # the subspace iteration along the loop
           E::SparseArrays.SparseMatrixCSC{Float64, Int} # the coarse matrix, m k + 1 + nshifts square
           F::Any                                        # its sparse QR factorisation
         tol::Float64                                    # the accuracy of the bases, as change
    maxsteps::Int                                        # subspace steps at most per update
           p::Int                                        # the directions whose accuracy is tested

    function Preconditioner(P, z::MultiVector; k::Int,
                                               tol::Real=1e-8,
                                               maxsteps::Int=100,
                                               p::Int=k)
        # ---- input checks ----
        1 <= p <= k ||
            throw(ArgumentError("p must be between 1 and k"))

        # ---- random bases at z, made accurate there ----
        S = Subspace(P, z; k=k)
        update!(S, z; tol=tol, maxsteps=maxsteps, p=p)

        # ---- the coarse matrix, from the bases before the last step ----
        M = new{typeof(S)}(S,
                           SparseArrays.spzeros(0, 0),
                           nothing,
                           tol,
                           maxsteps,
                           p)
        return _assemble!(M)
    end
end

"""
    update!(M::Preconditioner, z) -> M

Move `M` to the point `z`, as `nkroot!` of NKRoots.jl calls it at the start of every Newton
iteration: its subspace moved to `z`, with steps there from the bases of the point before until
they are accurate, at least one so that the images are those at `z`, and the coarse matrix
assembled and factorised from the bases before the last step.

# Arguments
- `M::Preconditioner`: the preconditioner, changed.
- `z::MultiVector`: the new point, kept, not copied.
"""
function NKRoots.update!(M::Preconditioner, z::MultiVector)
    # ---- the subspace at z, made accurate there ----
    update!(M.S, z; tol=M.tol, maxsteps=M.maxsteps, p=M.p)

    # ---- the coarse matrix, from the bases before the last step ----
    return _assemble!(M)
end

# E = Wᵀ A W from the bases before the last subspace step and the coordinates of their images
function _assemble!(M::Preconditioner)
    S  = M.S
    P  = S.P
    z  = S.z
    k  = length(S.U)
    m  = nsegments(z)
    ns = nshifts(z)
    n  = m*k + 1 + ns

    # ---- the rows and columns of seed i, and of the period and the shifts ----
    seed(i) = ((i - 1)*k + 1):(i*k)
    T       = m*k + 1
    sh(j)   = m*k + 1 + j

    # ---- the end points left by the subspace step, the end of the last segment shifted ----
    #      Xᵢ = φ_τ(uᵢ),   X_m = S(s) φ_τ(u_m)
    X = _copy(S.tmp)
    ns > 0 && shift!(P, _tail(X), _shifts(z))

    # ---- f there, each in the row of the seed it reaches, and the generators at the end of the
    #      last segment; f and the generators at the first seed ----
    #      Fᵢ₊₁ = f(φ_τ(uᵢ)),   F₁ = f(S(s) φ_τ(u_m)),   Gⱼ = gⱼ(S(s) φ_τ(u_m)),
    #      f₁ = f(u₁),   g₁ⱼ = gⱼ(u₁)
    F  = _roll!(rhs!(P, _similar(X), X))
    G  = [dshift!(P, _similar(_tail(X)), _tail(X), j) for j in 1:ns]
    f₁ = rhs!(P, _similar(_head(z.X)), _head(z.X))
    g₁ = [dshift!(P, _similar(_head(z.X)), _head(z.X), j) for j in 1:ns]

    # ---- the projections on the bases, one per seed, brought to the host ----
    #      Uᵢᵀ Fᵢ at every seed, f₁ᵀ U₁, U₁ᵀ Gⱼ and g₁ⱼᵀ U₁ at the first seed
    pf = [vec(Array(_dots(S.U[a], F))) for a in 1:k]
    ff = [_dot(f₁, _head(S.U[a])) for a in 1:k]
    pg = [_dot(_head(S.U[a]), G[j]) for a in 1:k, j in 1:ns]
    gg = [_dot(g₁[j], _head(S.U[a])) for a in 1:k, j in 1:ns]

    # ---- the entries ----
    Is = Int[]
    Js = Int[]
    Es = Float64[]
    entry!(r, c, e) = (push!(Is, r); push!(Js, c); push!(Es, e))

    for i in 1:m
        j = i % m + 1           # the seed segment i reaches

        # ---- Uⱼᵀ ψᵢ Uᵢ, -I, and the period column ----
        for a in 1:k, b in 1:k
            entry!(seed(j)[a], seed(i)[b], S.UᵀV[a, b, j])
        end
        for a in 1:k
            entry!(seed(i)[a], seed(i)[a], -1.0)
            entry!(seed(j)[a], T, pf[a][j]*z.w[i])
        end
    end

    # ---- the phase rows and the shift columns, at the first seed ----
    for a in 1:k
        entry!(T, seed(1)[a], ff[a])
        for j in 1:ns
            entry!(seed(1)[a], sh(j), pg[a, j])
            entry!(sh(j), seed(1)[a], gg[a, j])
        end
    end

    M.E = SparseArrays.sparse(Is, Js, Es, n, n)
    M.F = LinearAlgebra.qr(M.E)

    return M
end

# the coefficients Wᵀ v: the coordinates of every seed in its basis, direction by direction, then
# the period and the shifts
function _restrict(M::Preconditioner, v::MultiVector)
    k = length(M.S.U)
    c = [vec(Array(_dots(M.S.U[a], v.X))) for a in 1:k]

    # ---- seed by seed, as the rows of E ----
    return [vec(permutedims(reduce(hcat, c))); v.d]
end

"""
    ldiv!(out, M::Preconditioner, v) -> out

Minus `v` outside the span of the bases, and the solve with the coarse matrix on it,

    out = M⁻¹ v = -(I - W Wᵀ) v + W E⁻¹ Wᵀ v,

computed from the coefficients g = Wᵀ v and y = E⁻¹ g as

    out = -v + W (g + y)  on the seeds,   out = y  on the period and the shifts.

# Arguments
- `out::MultiVector`: overwritten with `M⁻¹ v`.
- `M::Preconditioner`: the preconditioner.
- `v::MultiVector`: the vector, not changed.
"""
function LinearAlgebra.ldiv!(out::MultiVector, M::Preconditioner, v::MultiVector)
    S = M.S
    k = length(S.U)
    m = nsegments(v)

    # ---- the coarse solve ----
    g = _restrict(M, v)
    y = M.F \ g

    # ---- the seeds: -v plus the bases with the coefficients g + y, one per seed ----
    #      outᵢ = -vᵢ + Σₐ (gᵢᵃ + yᵢᵃ) uᵢᵃ
    out .= .-v
    for a in 1:k
        c = _coefficients(v.X, [g[(i - 1)*k + a] + y[(i - 1)*k + a] for i in 1:m])
        out.X .+= c .* S.U[a]
    end

    # ---- the period and the shifts ----
    out.d .= y[(m*k + 1):end]

    return out
end

"""
    gram(M::Preconditioner, Q) -> R

The upper triangular `R` with `RᵀR` the Gram matrix of the vectors `M⁻¹ qⱼ`, from their
coefficients alone, as `nkroot!` of NKRoots.jl asks for it,

    RᵀR = (M⁻¹Q)ᵀ(M⁻¹Q) = I - ΓᵀΓ + ΘᵀΘ,   Γ = Wᵀ Q,   Θ = E⁻¹ Γ,

the two parts of each M⁻¹ qⱼ being orthogonal. `R` comes from a QR of `[L; Θ]` with
`LᵀL = I - ΓᵀΓ`, from the SVD of `Γ`, whose singular values are the cosines of the angles between
the vectors and the span of `W`, so that the lengths of `Θ` are not squared.

# Arguments
- `M::Preconditioner`: the preconditioner.
- `Q`: the orthonormal vectors `qⱼ`, a vector of [`MultiVector`](@ref)s, the basis of the
  Krylov space.
"""
function NKRoots.gram(M::Preconditioner, Q)
    # ---- the coefficients of the vectors and their coarse solves ----
    Γ = reduce(hcat, [_restrict(M, q) for q in Q])
    Θ = M.F \ Γ

    # ---- L with LᵀL = I - ΓᵀΓ, from the SVD of Γ ----
    F = LinearAlgebra.svd(Γ; full=true)
    c = [F.S; zeros(length(Q) - length(F.S))]
    L = sqrt.(max.(0.0, (1 .- c) .* (1 .+ c))) .* F.Vt

    # ---- the triangular factor of [L; Θ], without its Gram matrix ----
    return LinearAlgebra.UpperTriangular(LinearAlgebra.qr([L; Θ]).R)
end
