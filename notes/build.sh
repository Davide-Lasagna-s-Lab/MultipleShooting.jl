#!/bin/zsh
# Build the note with Tectonic, after the figures of scripts/ if asked,
#
#     ./build.sh            # build multiple-shooting.pdf
#     ./build.sh figures    # run scripts/compute.jl and scripts/plot.jl first
cd "$(dirname "$0")"
if [ "$1" = "figures" ]; then
    (cd scripts && julia --project=. compute.jl && julia --project=. plot.jl)
fi
tectonic --keep-logs multiple-shooting.tex 2>&1 | grep -E "error|Error|undefined" | head -30
grep -E "Overfull \\\\hbox \([0-9]{2,}|LaTeX Warning: (Reference|Citation|There were undefined)" multiple-shooting.log | head -30
