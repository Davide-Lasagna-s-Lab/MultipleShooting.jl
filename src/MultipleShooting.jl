# Multiple shooting for periodic orbits, matrix free: the unknowns, the residual and the product of
# its Jacobian with a vector, on top of a propagator that advances batches of states.
module MultipleShooting

import LinearAlgebra
import NKRoots
import Random
import SparseArrays

export propagate!,
       rhs!,
       tangent!,
       forced!,
       shift!,
       dshift!,
       dpropagate!,
       nsegments,
       states,
       similar_states,
       shift_forward!,
       shift_backward!,
       dots,
       coefficients,
       MultiVector,
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

include("seeds.jl")
include("interface.jl")
include("multivector.jl")
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
