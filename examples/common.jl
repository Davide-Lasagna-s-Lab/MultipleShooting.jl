# What the examples share: the look of the figures. Each example is a folder examples/<name> with
# its script <name>.jl and the figure that the script writes, and its script includes this file as
# include(joinpath(@__DIR__, "..", "common.jl")).
using CairoMakie
using Printf


# ---- the look of the figures, that of the examples of NKRoots.jl ----
const BLUE   = "#2a78d6"
const ORANGE = "#eb6834"
const AQUA   = "#1baf7a"
const YELLOW = "#eda100"
const INK    = "#52514e"
const SERIES = (BLUE, ORANGE, AQUA, YELLOW)

set_theme!(Theme(fontsize=14,
                 Axis=(xgridvisible=false,
                       ygridcolor=(:gray, 0.15),
                       topspinevisible=false,
                       rightspinevisible=false,
                       titlealign=:left,
                       titlefont=:regular,
                       titlesize=14),
                 Legend=(framevisible=false,),
                 Lines=(linewidth=2,),
                 Scatterlines=(linewidth=2, markersize=8)))

# save the figure next to the script that makes it, joinpath(@__DIR__, "name.png")
function finish(fig, path)
    save(path, fig; px_per_unit=2)
    return fig
end
