# The Jacobian of the residual, matrix free, with the products by the tangent equation, the
# variational equations of Sánchez et al. (2004) along every segment.
#
# With the reordered rows of residual.jl, the Jacobian at z applied to v = (v₁, …, v_m, δT, δs₁, …)
# has in the row of segment i + 1
#
#     ψᵢ vᵢ - vᵢ₊₁ + δτ f(φ_τ(uᵢ)),
#
# with ψᵢ the tangent map of segment i and δτ = δT/m the change of its length τ = T/m. In
# row 1, which closes the orbit, the end of the last segment is shifted, and the shifts add their
# columns,
#
#     S(s) ψ_m v_m - v₁ + δτ f(w) + Σⱼ δsⱼ gⱼ(w),   w = S(s) φ_τ(u_m),
#
# with gⱼ the generator of shift j, by the equivariance of the system. The scalar part holds the
# phase conditions, ⟨f(u₁), v₁⟩ in time and ⟨gⱼ(u₁), v₁⟩ in space, which keep the phase of the
# first seed. The product follows the residual step by step: the vector is advanced over the
# segments by the tangent equation, along the seeds in a batch of their own, so that a product
# integrates the seeds again; the end of the last segment is shifted, the columns of the period
# and of the shifts are added, and each row is moved into the row of the seed it has to reach
# before the vector is subtracted. f and the generators at the end points and at the first seed
# are kept, and computed again by update! at a new point, which integrates the seeds once.
#
# The end points come from the tangent L, which advances the seeds with the vectors, here with no
# perturbation, and the column of the period from the right-hand side F at the end points, the
# derivative of the exact flow, which differs from that of the integrator by the error of the
# scheme. The derivative of the flow dG, given, gives both, the derivative of the integrator, of
# which the residual is made.

"""
    Jacobian(L, F, z; S=nothing, dS=nothing, dG=nothing)

The Jacobian of the multiple-shooting residual at `z`, matrix free, with the tangent `L` and the
right-hand side `F`: `mul!(out, J, v)` or `J*v` applies it to v = (v₁, …, v_m, δT, δs₁, …, δs_c),

    (Jv)ᵢ₊₁ = ψᵢ vᵢ - vᵢ₊₁ + δτ ∂φ_τ(uᵢ)/∂τ,   i = 1, …, m - 1,
    (Jv)₁   = S(s) ψ_m v_m - v₁ + δτ S(s) ∂φ_τ(u_m)/∂τ + Σⱼ δsⱼ gⱼ(w),   w = S(s) φ_τ(u_m),
    (Jv)_T  = ⟨f(u₁), v₁⟩,   (Jv)_sⱼ = ⟨gⱼ(u₁), v₁⟩,

with δτ = δT/m the change of the length τ = T/m of a segment, or αᵢ δT for segments of unequal
lengths τᵢ = αᵢ T, the fractions of [`times`](@ref), ψᵢ the tangent maps, by the tangent equation
along the seeds, one integration per product, and the scalar rows the phase conditions. Building it integrates the seeds once, by the tangent with no perturbation, or by
`dG`, which also gives the column of the period.

# Arguments
- `L`: the tangent, called as `L(X, V, τ)` with one perturbation of every seed.
- `F`: the right-hand side, called as `F(dX, X)`.
- `z::ShootingVector`: the point, kept, not copied; `update!(J, z)` moves it to another.
- `S=nothing`, `dS=nothing`: the shift, called as `S(X, s...)`, and its generator, as
  `dS(dX, X, j)`, for relative periodic orbits.
- `dG=nothing`: the derivative of the flow in τ, called as `dG(X, V, τ)`, for the column of the
  period of the integrator, or `nothing` for `F` at the end points.
"""
mutable struct Jacobian{LT, FT, ST, DST, DGT, V<:ShootingVector, A, H}
       L::LT             # the tangent
       F::FT             # the right-hand side
       S::ST             # the shift, or nothing for periodic orbits
      dS::DST            # its generator, or nothing
      dG::DGT            # the derivative of the flow in τ, or nothing for F at the end points
       z::V              # the point, not a copy
     tmp::A              # work batch: the end points, then the seeds advanced with the vectors
    dXdT::A              # the column of the period: in row i, the derivative in T of the end of
                         # segment i, 1/m times that in τ, the last one shifted
      gₘ::Vector{H}      # the generators at the shifted end of the last segment, one per shift
       f::A              # f at the seeds, for the phase condition at the first: computed at all
                         # of them, so that F sees batches of m states only
      g₁::Vector{H}      # the generators at the first seed, one per shift

    function Jacobian(L, F, z::ShootingVector; S=nothing, dS=nothing, dG=nothing)
        # ---- input checks ----
        nshifts(z) > 0 && (S === nothing || dS === nothing) &&
            throw(ArgumentError("a relative periodic orbit needs its shift S and its generator dS"))

        # ---- the storage, filled by update! ----
        tmp  = similar(z.X)
        dXdT = similar(z.X)
        f    = similar(z.X)
        h    = similar(head(z.X))
        gₘ   = typeof(h)[similar(h) for _ in 1:nshifts(z)]
        g₁   = typeof(h)[similar(h) for _ in 1:nshifts(z)]

        J = new{typeof(L),
                typeof(F),
                typeof(S),
                typeof(dS),
                typeof(dG),
                typeof(z),
                typeof(f),
                typeof(h)}(L,
                           F,
                           S,
                           dS,
                           dG,
                           z,
                           tmp,
                           dXdT,
                           gₘ,
                           f,
                           g₁)
        return NKRoots.update!(J, z)
    end
end

"""
    update!(J::Jacobian, z) -> J
    update!(J::Jacobian, z, r) -> J

Move `J` to the point `z`: the column of the period, and f and the generators at the end points
of the segments and at the first seed, with one integration of the seeds. A method of
`NKRoots.update!`; the second form is the one `nkroot!` of NKRoots.jl calls.

# Arguments
- `J::Jacobian`: the Jacobian, changed.
- `z::ShootingVector`: the new point, kept, not copied.
- `r::ShootingVector`: the residual at `z`, not used.
"""
function NKRoots.update!(J::Jacobian, z::ShootingVector)
    m   = nsegments(z)
    τ   = times(z)
    J.z = z

    # ---- the end points, row i φ_τ(uᵢ), and their derivatives in τ, the column of the period:
    #      F at the end points, from the tangent with no perturbation, or those of the
    #      integrator, by the derivative of the flow ----
    copyto!(J.tmp, z.X)
    if J.dG === nothing
        J.dXdT .= 0
        J.L(J.tmp, J.dXdT, τ)
        J.F(J.dXdT, J.tmp)
    else
        J.dG(J.tmp, J.dXdT, τ)
    end

    # ---- each over m, since τ = T/m, or times its fraction, since τᵢ = wᵢ T ----
    if z.w === nothing
        J.dXdT ./= m
    else
        J.dXdT .*= coefficients(J.dXdT, z.w)
    end

    # ---- the end of the last segment shifted, and its derivative with it ----
    if nshifts(z) > 0
        J.S(tail(J.tmp), shifts(z)...)
        J.S(tail(J.dXdT), shifts(z)...)
    end

    # ---- f and the generators at the end of the last segment, and at the first seed: f at all
    #      the seeds, of which the phase condition takes the first ----
    J.F(J.f, z.X)
    for j in 1:nshifts(z)
        J.dS(J.gₘ[j], tail(J.tmp), j)
        J.dS(J.g₁[j], head(z.X), j)
    end

    return J
end

# as nkroot! of NKRoots.jl calls it, at the start of every Newton iteration
NKRoots.update!(J::Jacobian, z::ShootingVector, r::ShootingVector) = NKRoots.update!(J, z)

# out = J v. One integration of the seeds per product (the tangent L, in place on the device of
# the seeds); tmp is the only work batch, and the phase rows are dots on the host.
function LinearAlgebra.mul!(out::ShootingVector, J::Jacobian, v::ShootingVector)
    τ  = times(J.z)
    δT = period(v)

    # ---- the seeds and the vector ----
    #      tmpᵢ = uᵢ,   outᵢ = vᵢ
    copyto!(J.tmp, J.z.X)
    out .= v

    # ---- advanced over their segments by the tangent equation ----
    #      outᵢ = ψᵢ vᵢ
    J.L(J.tmp, out.X, τ)

    # ---- the end of the last segment shifted ----
    #      out_m = S(s) ψ_m v_m
    nshifts(J.z) > 0 && J.S(tail(out.X), shifts(J.z)...)

    # ---- plus the period column, and the shift columns at the end of the last segment ----
    #      outᵢ += δT Fᵢ, Fᵢ = (1/m) ∂φ_τ(uᵢ)/∂τ,   out_m += Σⱼ δsⱼ gⱼ(S(s) φ_τ(u_m))
    out.X .+= δT .* J.dXdT
    for j in 1:nshifts(J.z)
        tail(out.X) .+= shift(v, j) .* J.gₘ[j]
    end

    # ---- each row in the row of the seed it has to reach, minus the vector ----
    #      outᵢ₊₁ = ψᵢ vᵢ + δτ f(φ_τ(uᵢ)) - vᵢ₊₁,   out₁ = S(s) ψ_m v_m + … - v₁
    roll_forward!(out.X)
    out .= out .- v

    # ---- the phase conditions, in time and in space: dot of LinearAlgebra on the first states,
    #      which head takes as states of the type of the seeds, so that it is the inner product of
    #      the states, weighted when the type of the seeds has its own; the real part, for an
    #      array of complex states ----
    #      d₁ = ⟨f(u₁), v₁⟩,   d₁₊ⱼ = ⟨gⱼ(u₁), v₁⟩
    out.d[1] = real(LinearAlgebra.dot(head(J.f), head(v.X)))
    for j in 1:nshifts(J.z)
        out.d[1 + j] = real(LinearAlgebra.dot(J.g₁[j], head(v.X)))
    end

    return out
end

Base.:*(J::Jacobian, v::ShootingVector) = LinearAlgebra.mul!(similar(v), J, v)
