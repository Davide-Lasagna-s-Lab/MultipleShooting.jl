# Multiple shooting for periodic and relative periodic orbits, matrix free (Sánchez & Net 2010;
# Lasagna et al. 2026). Seeds u₁, …, u_m, m segments of length τ = T/m, period T, shifts
# s₁, …, s_c. The package holds the unknowns z = (u₁, …, u_m, T, s₁, …, s_c), the residual and the
# product of its Jacobian with a vector, a subspace iteration, a preconditioner on it, GMRES, and
# the searches in a parameter (sensitivity, continuation). It rests on callable objects of the
# caller, the flow G, the tangent L, the right-hand side F, the shift S, which advance batches of
# states (interface.jl).
module MultipleShooting

import LinearAlgebra
import NKRoots
import Random
import SparseArrays

export nsegments,
       states,
       head,
       tail,
       similar_states,
       roll_forward!,
       roll_backward!,
       dots,
       coefficients,
       Batched,
       Sliced,
       ShootingVector,
       period,
       nshifts,
       shift,
       shifts,
       Residual,
       Jacobian,
       JacobianFD,
       Subspace,
       iterate!,
       maxturn,
       Preconditioner,
       gmres,
       ritz_values,
       tangent_sensitivity,
       freesearch!,
       continuation!

include("interface.jl")
include("seeds/batched.jl")
include("seeds/sliced.jl")
include("shootingvector.jl")
include("residual.jl")
include("jacobian.jl")
include("jacobian_fd.jl")
include("subspace.jl")
include("preconditioner.jl")
include("gmres.jl")
include("sensitivity/sensitivity.jl")
include("continuation/parameter.jl")
include("continuation/continuation.jl")

end
