# Draws assets/logo.svg: an orbit, dashed, cut into four segments, each from its seed, a dot, to
# its end, an open circle a little off the next seed, the gap that multiple shooting closes.
#
#     julia assets/make_logo.jl

using Printf


# ---- the orbit, a closed curve, and its four seeds ----

# the point of the orbit at the phase θ, and the same pushed off it by δ along the normal
orbit(θ)    = (150.0*cos(θ) - 12.0*sin(2θ), 88.0*sin(θ) + 22.0*cos(2θ) - 6.0)
off(θ, δ)   = (orbit(θ)[1]*(1 + δ), orbit(θ)[2]*(1 + δ))

COLORS = ["#389826", "#4063d8", "#9558b2", "#cb3c33"]
seeds  = [π/2 - 2π*(i - 1)/4 + 0.6 for i in 1:4]

# the orbit itself, dashed, behind the segments
path(points) = "M" * join([@sprintf("%.1f %.1f", p...) for p in points], " L")
behind = path([orbit(θ) for θ in range(0, 2π; length=200)])

# segment i, from seed i to a little off seed i + 1, bending away from the orbit as an error grows
function segment(i)
    a = seeds[i]
    b = a - 2π/4
    θ = range(a, b; length=60)
    δ = [0.16*((a - t)/(a - b))^2 for t in θ]

    return path([off(t, d) for (t, d) in zip(θ, δ)]), off(b, 0.16)
end

segments = [segment(i) for i in 1:4]

curves = [@sprintf("<path d=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"6\" stroke-linecap=\"round\"/>",
                   s[1], c) for (s, c) in zip(segments, COLORS)]
ends   = [@sprintf("<circle cx=\"%.1f\" cy=\"%.1f\" r=\"7\" fill=\"#ffffff\" stroke=\"%s\" stroke-width=\"3.5\"/>",
                   s[2]..., c) for (s, c) in zip(segments, COLORS)]
dots   = [@sprintf("<circle cx=\"%.1f\" cy=\"%.1f\" r=\"7.5\" fill=\"#242936\"/>", orbit(θ)...) for θ in seeds]


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 300" role="img" aria-labelledby="title desc">
  <title id="title">MultipleShooting.jl logo</title>
  <desc id="desc">An orbit, dashed, cut into four segments in Julia colors, each from its seed, a dot, to its end, an open circle a little off the next seed, beside the MultipleShooting.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.10"/>
    </filter>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="912" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- Left illustration: the orbit and its segments -->
  <g transform="translate(222 150)">
    <path d="$behind Z" fill="none" stroke="#c6ccd6" stroke-width="2.6" stroke-dasharray="8 7"/>
    <g filter="url(#soft)">
      $(join(curves, "\n      "))
      $(join(ends, "\n      "))
      $(join(dots, "\n      "))
    </g>
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(430 96)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="48" font-weight="820" fill="#222734" letter-spacing="0">MultipleShooting.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Periodic orbits by multiple shooting,</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">matrix free, on batches of states.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
