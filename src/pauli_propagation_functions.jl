module pauli_propagation_functions
export decode_pauli, compute_matrix, overlap, pauli_norm, shannon_entropy, renyi_entropy, applynoiselayer, truncate_max_size, propagate_layerbylayer

using PauliPropagation
using PauliPropagation: Xmat, Ymat, Zmat
using LinearAlgebra

#------------ Decode Pauli string ------------
function decode_pauli(pauli_string::Unsigned, num_qubits::Int)::String
  raw"""
  decode_pauli(pauli_string::Unsigned, num_qubits::Int)::String

  Convert a bit-encoded Pauli string into its human-readable string representation.

  ### Arguments

  * `pauli_string`: An unsigned integer representing the Pauli string, where each qubit is encoded by two bits ($00 \to I$, $01 \to X$, $10 \to Y$, $11 \to Z$).
  * `num_qubits`: The total number of qubits represented in the string.

  ### Returns

  * A `String` containing the sequence of Pauli operators (e.g., `"IXYZ"`).

  ### Notes

  The function supports any standard Julia `Unsigned` integer type (e.g., `UInt8`, `UInt64`). Using `Unsigned` in the function signature is the recommended idiomatic approach to ensure type safety while maintaining flexibility.
  """
  mapping = Dict(0b00 => "I", 0b01 => "X", 0b10 => "Y", 0b11 => "Z")
    
  res = ""
  for i in 0:(num_qubits - 1)
    bits = (pauli_string >> (2 * i)) & 0b11
    res *= mapping[bits]
  end
  return res
end

#------------ PauliSum -> Matrix ------------
function compute_matrix(observable::Union{PauliSum, PauliString})::Matrix{ComplexF64}
  raw"""
  compute_matrix(observable::Union{PauliSum, PauliString})::Matrix{ComplexF64}

  Construct the dense matrix representation of a Pauli operator or sum of Pauli operators.

  ### Arguments

  * `observable`: A `PauliSum` or `PauliString` object to be converted into a matrix.

  ### Returns

  * A `Matrix{ComplexF64}` of size $2^n \times 2^n$, where $n$ is the number of qubits.

  ### Notes

  This function iterates through the Pauli strings within the observable, constructs the corresponding Kronecker product of basis matrices, and sums them weighted by their coefficients. It assumes the existence of `Xmat`, `Ymat`, and `Zmat` as globally defined basis matrices.
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end
  nqubits = observable.nqubits
  mapping = Dict('I' => I(2), 'X' => Xmat, 'Y' => Ymat, 'Z' => Zmat)

  result = zeros(ComplexF64, 2^nqubits, 2^nqubits)
  for (pauli_string, coeff) in observable
    string = decode_pauli(pauli_string, nqubits)
    result_string = [1.0 + 0.0im;;]
    for op in string
      result_string = kron(result_string, mapping[op])
    end
    result += coeff * result_string
  end
  return result
end

#------------ overlap ------------ 
function overlap(observable::Union{PauliSum, PauliString}, ψ::Union{Vector{Float64}, Vector{Int64}})::Float64 
  raw"""
  overlap(observable::Union{PauliSum, PauliString}, ψ::Union{Vector{Float64}, Vector{Int64}})::Float64

  Compute the expectation value $\langle \psi | \hat{O} | \psi \rangle$ of a Pauli observable for a given state vector.

  ### Arguments

  * `observable`: A `PauliSum` or `PauliString` representing the operator $\hat{O}$.
  * `ψ`: A real-valued state vector (`Vector{Float64}`) of length $2^n$.

  ### Returns

  * The expectation value as a `Float64`.

  ### Notes

  This function calculates the expectation value by summing the contributions of individual Pauli strings. If `ψ` is the computational basis state $|0\dots0\rangle$, it optimizes the calculation by calling `overlapwithzero`. 
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end

  nqubits = observable.nqubits
  mapping = Dict('I' => I(nqubits), 'X' => Xmat, 'Y' => Ymat, 'Z' => Zmat)
  
  state0 = append!([1],[0 for _ in 2:(2^nqubits)])
  if ψ == state0
    return overlapwithzero(observable)
  end 

  result = 0.0
  for (pauli_string, coeff) in observable
    string = decode_pauli(pauli_string, nqubits)
    result_string = [1.0 + 0.0im;;]
    for op in string
      result_string = kron(result_string, mapping[op])
    end
    result += coeff * ψ' * result_string * ψ
  end
  return result
end

#------------ Pauli Norm ------------ 
function pauli_norm(observable::Union{PauliSum, PauliString})::Float64
  raw"""
  pauli_norm(observable::Union{PauliSum, PauliString})::Float64

  Calculate the squared Frobenius norm (sum of squared coefficients) of a Pauli operator or sum of Pauli operators.

  ### Arguments

  * `observable`: A `PauliSum` or `PauliString` object.

  ### Returns

  * The sum of the squared absolute values of the coefficients as a `Float64`.

  ### Notes

  This function computes $\sum_i |c_i|^2$ for an observable $\hat{O} = \sum_i c_i P_i$. This is useful for diagnostics related to the magnitude of the operator in the Pauli basis.
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end
  return sum(((P, c),) -> abs(c)^2, observable; init=0.0)
end

#------------ Pauli Entropy ------------ 
function shannon_entropy(observable::Union{PauliSum, PauliString})::Float64
  raw"""
  shannon_entropy(observable::Union{PauliSum, PauliString})::Float64

  Calculate the Shannon entropy of the squared coefficients of the Pauli observable.

  ### Arguments

  * `observable`: A `PauliSum` or `PauliString` object.

  ### Returns

  * The Shannon entropy $S = -\sum_i |c_i|^2 \ln(|c_i|^2)$ as a `Float64`.

  ### Notes

  This measure treats the normalized squared coefficients as a probability distribution, providing a diagnostic for the sparsity or distribution of the operator across the Pauli basis. It assumes the coefficients have been normalized such that $\sum |c_i|^2 = 1$.
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end
  return sum(((P, c),) -> c != 0 ? -abs(c)^2 * log(abs(c)^2) : 0.0, observable; init=0.0)
end

function renyi_entropy(observable::Union{PauliSum, PauliString{}}; k::Int64=2)::Float64
  raw"""
  renyi_entropy(observable::Union{PauliSum, PauliString}; k::Int64=2)::Float64

  Calculate the Rényi entropy of order `k` for the Pauli weight distribution of an `observable`.

  This metric quantifies the "spread" or complexity of an operator in the Pauli basis. An entropy of 0 indicates that the observable is a single Pauli string, while higher values indicate that the observable is fragmented into many terms.

  ### Arguments

  * `observable`: The `PauliSum` or `PauliString` to analyze.
  * `k`: The order of the Rényi entropy (default is 2).

  ### Returns

  * `Float64`: The entropy value $M_k = \frac{1}{1-k} \ln \sum_P \pi(P)^k$, where $\pi(P)$ is the normalized squared weight of each Pauli string.

  ### Notes

  For $k=1$, the function automatically returns the result of `shannon_entropy`. The function assumes coefficients are normalized such that the sum of squared magnitudes represents a probability distribution.
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end

  if k == 1 # L'entropie de Shannon (k=1) est la limite de Rényi
    return shannon_entropy(observable)
  end

  norm_sq = pauli_norm(observable) # compute \sum_P |c_P|²  
  if norm_sq == 0
    return 0.
  end

  sum_coeffs_2k = sum(p -> abs(p.second)^(2*k), observable; init=0.0) # p = (P, c_P)
  zeta_k = sum_coeffs_2k / (norm_sq^k)

  return log(zeta_k) / (1-k)
end

#------------ Noise layer ------------
function applynoiselayer(psum::PauliSum;depol_strength=0.02, dephase_strength=0.02, noise_level=1.0)
  raw"""
  applynoiselayer(psum::PauliSum; depol_strength::Float64=0.02, dephase_strength::Float64=0.02, noise_level::Float64=1.0)

  Apply a noise layer to a `PauliSum` by attenuating its coefficients in-place.

  This function simulates decoherence by scaling each Pauli string's coefficient based on its Hamming weight and the count of specific operator types.

  ### Arguments

  * `psum`: The `PauliSum` object to be modified.
  * `depol_strength`: The base rate of depolarizing noise (affects all non-identity operators).
  * `dephase_strength`: The base rate of dephasing noise (specifically affects X and Y operators).
  * `noise_level`: A global multiplier to scale the overall noise intensity.

  ### Mathematical Model

  For each Pauli string $P$, the coefficient $c_P$ is updated as:
  $c_P \leftarrow c_P \cdot (1 - \text{noise\_level} \cdot \text{depol\_strength})^{\text{weight}(P)} \cdot (1 - \text{noise\_level} \cdot \text{dephase\_strength})^{\text{count}_{XY}(P)}$
  """
  for (pstr, coeff) in psum
    set!(psum, pstr,
         coeff*(1-noise_level*depol_strength)^countweight(pstr)*(1-noise_level*dephase_strength)^countxy(pstr))
  end
end

#------------ Propagate Layer by layer ------------ 
function get_layer(
  layer_idx::Integer,
  ngate_bylayer::Integer, 
  circuit::Union{Gate, Vector{Gate}}, 
  parameters::Union{Nothing, Vector{Float64}}
  )::Tuple{Union{Gate, Vector{Gate}}, Union{Nothing, Vector{Float64}}}
  raw"""
  get_layer(layer_idx::Integer, ngate_bylayer::Integer, circuit::Union{Gate, Vector{Gate}}, parameters::Union{Nothing, Vector{Float64}})::Tuple{Vector{Gate}, Union{Nothing, Vector{Float64}}}

  Extract a specific layer of gates and their corresponding parameters from a quantum circuit.

  ### Arguments

  * `layer_idx`: The 1-based index of the layer to retrieve.
  * `ngate_bylayer`: The number of gates contained within each layer.
  * `circuit`: A `Vector` of `Gate` objects representing the full circuit.
  * `parameters`: A vector of gate parameters (e.g., rotation angles) or `nothing`.

  ### Returns

  * A tuple containing the slice of the `circuit` vector corresponding to the requested layer, and the corresponding slice of `parameters` (or `nothing`).


  ### Notes

  The function assumes the `circuit` and `parameters` are structured linearly, with gates and parameters organized in contiguous blocks of size `ngate_bylayer`.
  """
  first_gate_idx = ((layer_idx-1)*ngate_bylayer)+1; last_gate_idx = (layer_idx * ngate_bylayer)
  layer_gates = circuit[first_gate_idx:last_gate_idx]

  if parameters === nothing
    parameter = nothing
  else
    parameter = parameters[first_gate_idx:last_gate_idx]
  end
  return layer_gates, parameter
end

function truncate_max_size!(psum::PauliSum, max_size::Int64)::PauliSum
  """
  truncate_max_size!(psum::PauliSum, max_size::Int64)::PauliSum

  Truncate `psum.terms` in-place to keep only the `max_size` dominant terms.

  `psum.terms` is a `Dict{UInt*, Float64}` mapping each Pauli string to its
  coefficient. If the number of terms exceeds `max_size`, the terms with the
  smallest coefficients are discarded, retaining only the `max_size` largest.

  # Arguments
  - `psum::PauliSum`: the operator whose terms are truncated in-place.
  - `max_size::Int64`: maximum number of Pauli string / coefficient pairs to retain.

  # Returns
  The modified `psum::PauliSum` with at most `max_size` terms.

  # Notes
  - No-op if `length(psum.terms) <= max_size`.
  - The dictionary is mutated directly; the top `max_size` terms by coefficient
    magnitude are preserved, all others are dropped.
  """
  dict = psum.terms
  n = length(dict)
    n <= max_size && return psum

    Pc_pairs = collect(dict) # Vector(PauliString => Coeff)
    partialsort!(Pc_pairs, 1:max_size, by=last, rev=true) # 'by=last' compare the Coeff, '1:max_size' not all of them, just the top "max_size" ones, 'rev=true' sort them.

    empty!(dict)
    sizehint!(dict, max_size)
    @inbounds for i in 1:max_size
        P, c = Pc_pairs[i]  
        dict[P] = c
    end
    return psum
end

function propagate_1layer(
  layer_gates::Union{Gate, Vector{Gate}}, 
  current::Union{PauliSum, PauliString},
  parameter::Union{Nothing, Float64, Vector{Float64}};
  max_weight::Union{Nothing, Integer}=nothing,
  max_size::Union{Nothing, Integer}=nothing,
  min_abs_coeff::Float64=0.,
  γ::Float64=0., # for the Noise
  normalize::Bool=true
  )::PauliSum
  raw"""
  propagate_1layer(layer_gates::Union{Gate, Vector{Gate}}, current::Union{PauliSum, PauliString}, parameter::Union{Nothing, Float64, Vector{Float64}}; max_weight::Union{Nothing, Integer}=nothing, min_abs_coeff::Float64=0.0, γ::Float64=0.0)::PauliSum

  Propagate a Pauli observable through a single layer of a quantum circuit, including optional noise and renormalization.

  ### Arguments

  * `layer_gates`: The gate(s) in the current layer.
  * `current`: The input `PauliSum` or `PauliString` to propagate.
  * `parameter`: Parameters associated with the gates, if any.
  * `max_weight`: Optional truncation parameter to limit the Pauli string weight.
  * `min_abs_coeff`: Minimum absolute coefficient value for truncation.
  * `γ`: Noise intensity parameter for depolarizing noise.
  * `normalize` : If `true`, it renormalizes the observable

  ### Returns

  * A `PauliSum` object representing the observable after propagation, noise application, and normalization.


  ### Notes

  This function propagates the observable using `propagate` in the PauliPropagation package, applies a depolarizing noise layer if `γ > 0`, and renormalizes the observable such that the sum of the squared coefficients equals 1.
  """
  if typeof(current) <: PauliString{<:Unsigned, Float64}
    current = PauliSum(current)
  end

  if max_weight === nothing
    max_weight = current.nqubits 
  end

  current = PauliPropagation.propagate(layer_gates, current, parameter; max_weight, min_abs_coeff)

  if max_size !== nothing
    truncate_max_size!(current, max_size)
  end 

  if !(γ == 0.)
	  applynoiselayer(current; depol_strength=1, dephase_strength=0, noise_level=γ)
  end
  if normalize
    current /= sqrt(pauli_norm(current)) # on divise par la norm pour que \sum |c_\alpha|²=1 malgres les troncations
  end
  return current
end

function propagate(
  circuit::Union{Gate, Vector{Gate}}, 
  observable::Union{PauliSum, PauliString}, 
  nlayers::Int64, 
  parameters::Union{Vector{Float64}, Nothing}=nothing; 
  max_weight::Union{Integer,Nothing}=nothing, 
  max_size::Union{Integer,Nothing}=nothing, 
  min_abs_coeff::Float64=0.,
  k::Union{Int64, Nothing}=nothing, # for the Entropy
  ψ0::Union{Vector{Float64}, Nothing}=nothing, # for the Overlap
  γ::Float64=0., # for the Noise
  normalize::Bool=true,
  disable_print::Bool=false
  )::Tuple{PauliSum, Dict{String, Any}}
  raw"""
  propagate(circuit::Union{Gate, Vector{Gate}}, observable::Union{PauliSum, PauliString}, nlayers::Int64, parameters::Union{Vector{Float64}, Nothing}=nothing; max_weight::Union{Integer, Nothing}=nothing, min_abs_coeff::Float64=0.0, k::Union{Int64, Nothing}=nothing, ψ0::Union{Vector{Float64}, Nothing}=nothing, γ::Float64=0.0, disable_print::Bool=false)::Tuple{PauliSum, Dict{String, Any}}

  Propagate a Pauli observable through a quantum circuit layer-by-layer in the Heisenberg picture.

  ### Arguments

  * `circuit`: The sequence of gates representing the quantum circuit.
  * `observable`: The initial `PauliSum` or `PauliString` to propagate.
  * `nlayers`: The number of layers in the circuit.
  * `parameters`: Optional vector of gate parameters.
  * `max_weight`: Optional truncation parameter for Pauli string weight.
  * `min_abs_coeff`: Minimum absolute coefficient value for truncation.
  * `k`: Optional order for Rényi entropy calculation.
  * `ψ0`: Optional reference state vector for overlap calculation.
  * `γ`: Noise intensity parameter for depolarizing noise.
  * `normalize` : If `true`, at each layer, it renormalizes the observable
  * `disable_print`: If `true`, suppresses progress output.

  ### Returns

  * A `Tuple` containing:
  * `current`: The final `PauliSum` after propagation.
  * `result`: A `Dict` containing the collected diagnostics (`"norm"`, `"overlap"`, `"S"`, and `"time"`).


  ### Notes

  In the Heisenberg picture, the circuit is processed in reverse order (from `nlayers` down to 1). If diagnostics (`k` or `ψ0`) are provided, the corresponding metrics are tracked at each layer and returned in the result dictionary.
  """
  t = time()
  ngate_bylayer = size(circuit,1) ÷ nlayers

  overlaps, entropy, norm = Float64[], Float64[], Float64[]

  current = PauliPropagation.PauliSum(observable)

  if k != nothing
    push!(entropy, renyi_entropy(current; k))
  end
  if ψ0 != nothing
    push!(overlaps, overlap(current, ψ0))
  end
  push!(norm, pauli_norm(current))
  
  for layer_idx in nlayers:-1:1 # pour propager on a besoin de donner les couches dans le sens inverse /!\ (Heisenberg picture)
    layer_gates, parameter = get_layer(layer_idx, ngate_bylayer, circuit, parameters)
    current = propagate_1layer(layer_gates, current, parameter; max_weight, max_size, min_abs_coeff, γ, normalize)
    if k != nothing
      push!(entropy, renyi_entropy(current; k))
    end
    if ψ0 != nothing
      push!(overlaps, overlap(current, ψ0))
    end
    push!(norm, pauli_norm(current))

    step=nlayers-layer_idx+1
    if step % max(1, nlayers÷10)==0 && !disable_print
      println("layer : ",step,"/",nlayers," complete")
    end
  end

  elapsed_time = time() - t
  println("Time taken by pauli_propagation_functions.propagate: ", elapsed_time, " seconds")

  result = Dict("norm" => norm, "overlap" => overlaps, "S" => entropy, "time" => elapsed_time)
 
  return current, result
end

#------------ Find optimal truncations ------------ 
function find_truncations(
  tolerance::Float64, 
  circuit::Union{Gate, Vector{Gate}}, 
  observable::Union{PauliSum, PauliString}, 
  nlayers::Int64, 
  parameters::Union{Nothing, Vector{Float64}}=nothing;
  γ::Float64=0., # for the Noise
  disable_print::Bool=false
  )::Tuple{Int64, Float64}
  raw"""
  find_truncations(tolerance::Float64, circuit::Union{Gate, Vector{Gate}}, observable::Union{PauliSum, PauliString}, nlayers::Int64, parameters::Union{Nothing, Vector{Float64}}=nothing; γ::Float64=0.0, disable_print::Bool=false)::Tuple{Int64, Float64}

  Determine the optimal truncation parameters (`max_weight` and `min_abs_coeff`) for the Pauli propagation method.

  ### Arguments

  * `tolerance`: The relative tolerance used to check for convergence of the overlap sequence.
  * `circuit`: The sequence of gates representing the quantum circuit.
  * `observable`: The initial `PauliSum` or `PauliString` to propagate.
  * `nlayers`: The number of layers in the circuit.
  * `parameters`: Optional vector of gate parameters.
  * `γ`: Noise intensity parameter for depolarizing noise.
  * `disable_print`: If `true`, suppresses progress output.

  ### Returns

  * A `Tuple` containing:
  * `max_weight`: The determined optimal maximum weight for Pauli strings.
  * `min_abs_coeff`: The determined optimal minimum absolute coefficient threshold.


  ### Notes

  This function iteratively performs a sensitivity analysis by propagating the observable through the circuit, increasing `max_weight` and then decreasing `min_abs_coeff` until the relative change in the overlap sequence falls within the specified `tolerance`.
  """
  nqubits = observable.nqubits
  ngate_bylayer = size(circuit,1) ÷ nlayers
  ψ0 = append!([1],[0 for _ in 2:(2^nqubits)]) # |0> state
  
  smaller_min_abs_coeff = 1e-13
  
  heisenberg_circuit, parameter_list = Vector{Gate}[], Union{Float64, Vector{Float64},Nothing}[]
  for layer_idx in nlayers:-1:1
    layer_gates, parameter = get_layer(layer_idx, ngate_bylayer, circuit, parameters)
    push!(heisenberg_circuit, layer_gates)
    push!(parameter_list, parameter)
  end

  # ----- Max weight TEST -----
  max_weight = 2 #+1 for the first test

  overlaps = fill(Inf, nlayers)

  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    max_weight += max(1, nqubits÷10)

    if max_weight >= nqubits
      max_weight = nqubits
      break
    end

    current = observable
    overlaps = Float64[]
    for i in 1:nlayers
      current = propagate_1layer(heisenberg_circuit[i], current, parameter_list[i]; max_weight, min_abs_coeff=smaller_min_abs_coeff, γ)
      push!(overlaps, overlap(current, ψ0))
    end
    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end

  # ----- Min abs coeff TEST -----
  min_abs_coeff_power = -1 #-1 for the first test

  overlaps = fill(Inf, nlayers)
  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    min_abs_coeff_power -= 1
    min_abs_coeff = 10^float(min_abs_coeff_power)

    if min_abs_coeff <= smaller_min_abs_coeff
      if !disable_print
        println("Optimal truncations find : (Max weight=$max_weight, Min abs coeff=1e$smaller_min_abs_coeff)")
      end
      return max_weight, smaller_min_abs_coeff
    end

    current = observable
    overlaps = Float64[]
    for i in 1:nlayers
      current = propagate_1layer(heisenberg_circuit[i], current, parameter_list[i]; max_weight, min_abs_coeff, γ)
      push!(overlaps, overlap(current, ψ0))
    end
    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end
  min_abs_coeff = 10^float(min_abs_coeff_power)
  if !disable_print
    println("Optimal truncations find : (Max weight=$max_weight, Min abs coeff=1e$min_abs_coeff_power)")
  end
  return max_weight, min_abs_coeff
end

end # module