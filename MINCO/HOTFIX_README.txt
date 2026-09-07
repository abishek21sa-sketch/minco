MINCO Phase 2 Julia Manifest Hotfix v0.8.0a3

Merge these files into the existing MINCO root and replace matching files.
This patch adds Pkg.resolve() before Pkg.instantiate() so a stale Julia Manifest.toml
from an earlier MINCO phase is safely reconciled with the current Project.toml.
