# Multiple shooting for periodic orbits, matrix free: the unknowns, the residual and the product of
# its Jacobian with a vector, on top of a propagator that advances batches of states.
module MultipleShootingV2

using LinearAlgebra
using Random

import NKRoots: update!

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
       update!,
       Subspace,
       iterate!,
       change,
       tangent_sensitivity

include("batch.jl")
include("interface.jl")
include("multivector.jl")
include("residual.jl")
include("jacobian.jl")
include("subspace.jl")
include("gmres.jl")
include("sensitivity.jl")

end
