# Multiple shooting for periodic orbits, matrix free: the unknowns, the residual and the product of
# its Jacobian with a vector, on top of a flow and its tangent that advance batches of states.
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
