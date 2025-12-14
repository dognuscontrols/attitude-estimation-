using Pkg
Pkg.activate(".")
Pkg.add([
  "LinearAlgebra", "Random", "Statistics",
  "JuMP", "SCS", "OSQP",
  "SparseArrays", "Plots", "MosekTools", "StaticArrays", "DelimitedFiles")
])
