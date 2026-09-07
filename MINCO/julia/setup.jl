import Pkg
Pkg.activate(@__DIR__)
# Julia installations created by juliaup may have no package registry yet.
# Bootstrap the public registry before resolving the pinned project.
if isempty(Pkg.Registry.reachable_registries())
    Pkg.Registry.add("General")
end
# Resolve first so merge-in upgrades cannot retain a stale Manifest.toml
# whose dependency graph predates the current Project.toml.
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
println("MINCO Julia optimization environment ready.")
