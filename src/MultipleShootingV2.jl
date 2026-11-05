# Multiple shooting for periodic orbits, matrix free: the unknowns, the residual and the product of
# its Jacobian with a vector, on top of a propagator that advances batches of states.
module MultipleShootingV2

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
       Batched,
       Sliced,
       MultiVector,
       nsegments,
       period,
       nshifts,
       shift,
       Residual,
       Jacobian,
       Subspace,
       iterate!,
       change,
       Preconditioner,
       gmres,
       ritz_values,
       tangent_sensitivity

include("batch.jl")
include("interface.jl")
include("multivector.jl")
include("residual.jl")
include("jacobian.jl")
include("subspace.jl")
include("preconditioner.jl")
include("gmres.jl")
include("sensitivity.jl")

end
