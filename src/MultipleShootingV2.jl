# Multiple shooting for periodic orbits, matrix free: the unknowns, the residual and the product of
# its Jacobian with a vector, on top of a propagator that advances batches of states.
module MultipleShootingV2

using LinearAlgebra

import NKRoots: update!

export propagate!,
       rhs!,
       tangent!,
       shift!,
       dshift!,
       MultiVector,
       nsegments,
       period,
       nshifts,
       shift,
       Residual,
       Jacobian,
       update!

include("interface.jl")
include("batch.jl")
include("multivector.jl")
include("residual.jl")
include("jacobian.jl")

end
