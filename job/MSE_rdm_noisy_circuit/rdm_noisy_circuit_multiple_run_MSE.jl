# --- Import ---
using Pkg; Pkg.add("LaTeXStrings"); Pkg.add("PauliPropagation"); Pkg.add("ITensors"); Pkg.add("ITensorMPS"); Pkg.add("DataFrames"); Pkg.add("CSV")

# Pauli Propagation
using PauliPropagation

include("../src/pauli_propagation_functions.jl")
import .pauli_propagation_functions as pp

# MPO
using ITensors, ITensorMPS

include("../src/matrix_product_operator_functions.jl")
import .mpo_functions as mpo

# exact
include("../src/exact_functions.jl")
import .exact_functions as ext

# other
using DataFrames, CSV

include("../src/circuit.jl")
import .circuit as ct

include("../src/utils.jl")
import .utils as us

import Statistics: mean

# --- Parameters ---
run = parse(Int64, ARGS[1])
path = joinpath("results", "run_$run")
mkpath(path)

# Observable : Z_i
i = 2

# qubits list
Ns = [7, 9]

# Gamma list = lambda_list/Nqubits
lambda_list = 0:0.03:0.4

normalize = false

for nqubits in Ns
  nlayers = nqubits

  power_list = (nqubits-5):(nqubits-1)
  maxdim_list = 2 .^ (power_list)
  maxsize_list = 4 .^(power_list)
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  circuit_rdm = ct.random_circuit(nqubits, nlayers; separateOddEvenLayer=true)

  # define initial state |0> state
  ψ0_exact = append!([1.],[0. for _ in 2:(2^nqubits)])

  ψ0_pp = ψ0_exact

  ψ0_mps = MPS(circuit_rdm["sites"], "0")

  # define observable Z_i = I...IZI...I
  Z_i_exact = ext.get_Zi(nqubits, i)

  Z_i_pp = PauliString(nqubits, :Z, i)

  ops = ["Id" for n in 1:nqubits]
  ops[i] = "Z"
  Z_i_mpo = MPO(circuit_rdm["sites"], ops)

  # --- PROPAGATION ---
  error_mpo_dict = Dict("N_qubits" => nqubits, "nlayer" => 0:(nlayers*2))
  error_pp_dict = Dict("N_qubits" => nqubits, "nlayer" => 0:(nlayers*2))

  for lambda in lambda_list
    gamma = lambda/nqubits
    println("---------- gamma=$lambda / $nqubits ----------")

    error_pp_list, error_mpo_list = Vector{Float64}[], Vector{Float64}[]
    exact_value_sq = Vector{Float64}[]

    println("------- gamma=$lambda / $nqubits, Exact -------")
    Zi_t_exact, result_exact = ext.propagate_layerbylayer(circuit_rdm["exact"], Z_i_exact; ψ0=ψ0_exact, γ=gamma, normalize)
    overlap_exact = result_exact["overlap"]

    for (maxdim, max_size) in zip(maxdim_list, maxsize_list)
      println("------- gamma=$lambda / $nqubits, Pauli, max_size=$max_size -------")
      psum, result_pp = pp.propagate_layerbylayer(circuit_rdm["pauli"], Z_i_pp, nlayers*2; max_size, ψ0=ψ0_pp, γ=gamma, normalize)

      println("------- gamma=$lambda / $nqubits, MPO, maxdim=$maxdim -------")
      Z_it_mpo, result_mpo = mpo.propagate_layerbylayer(circuit_rdm["mpo"], Z_i_mpo; maxdim, ψ0=ψ0_mps, γ=gamma, normalize)

      # --- Save Data ---
      error_pp_dict["sq error, gammaN=$lambda, maxsize=$max_size"] = @. abs(overlap_exact - result_pp["overlap"])^2
      error_mpo_dict["sq error, gammaN=$lambda, maxdim=$maxdim"] = @. abs(overlap_exact - result_mpo["overlap"])^2
    end
    error_pp_dict["exact value sq, gammaN=$lambda"] = overlap_exact.^2 # pour l'erreur relative
    error_mpo_dict["exact value sq, gammaN=$lambda"] = overlap_exact.^2
  end

  # --- Save Data ---
  complete_path_error_mpo = joinpath(path, "results_N_$(nqubits)-MSE_mpo.csv")
  complete_path_error_pp = joinpath(path, "results_N_$(nqubits)-MSE_pp.csv")

  error_mpo_df = DataFrame(error_mpo_dict)
  error_pp_df = DataFrame(error_pp_dict)

  CSV.write(complete_path_error_mpo, error_mpo_df)
  CSV.write(complete_path_error_pp, error_pp_df)
end