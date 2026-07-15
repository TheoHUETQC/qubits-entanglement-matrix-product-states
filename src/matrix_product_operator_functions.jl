module mpo_functions
export compute_matrix, overlap, operator_entropy, applynoiselayer, propagate_layerbylayer, find_truncations

using ITensors, ITensorMPS

#------------ MPO -> Matrix ------------
function compute_matrix(mpo::MPO, sites::Vector{<:Index})::Matrix{ComplexF64}
  raw"""
  compute_matrix(mpo::MPO, sites::Vector{<:Index})::Matrix{ComplexF64}

  Construct the dense matrix representation of a Matrix Product Operator (MPO).

  ### Arguments

  * `mpo`: The `ITensors.MPO` object to be converted.
  * `sites`: A vector of `ITensors.Index` objects representing the sites associated with the MPO.

  ### Returns

  * A `Matrix{ComplexF64}` of size $2^n \times 2^n$, where $n$ is the number of sites.

  ### Notes

  This function contracts the MPO into a single ITensor and reshapes it into a dense matrix. It temporarily increases the `ITensors` warning order to prevent order-related warnings during the contraction of large networks. The function assumes the sites are ordered correctly for the conversion.
  """
  ITensors.set_warn_order(20) # Increase the warning threshold to avoid the ITensor order warning
  nqubits = length(sites)

  tensor = contract(mpo) # Contract the MPO into a single ITensor

  row_indices = [sites[j]' for j in reverse(1:nqubits)]
  col_indices = [sites[j]  for j in reverse(1:nqubits)]
    
  tensor_array = Array(tensor, row_indices..., col_indices...)

  dim = 2^nqubits
  ITensors.reset_warn_order()
  return reshape(tensor_array, dim, dim)
end

#------------ Overlap with a state psi ------------
function overlap(O::MPO, ψ::MPS)::Float64
  raw"""
  overlap(O::MPO, ψ::MPS)::Float64

  Compute the expectation value $\langle \psi | \hat{O} | \psi \rangle$ of an MPO observable for a given Matrix Product State (MPS).

  ### Arguments

  * `O`: The `MPO` operator to evaluate.
  * `ψ`: The `MPS` representing the quantum state.

  ### Returns

  * The real-valued expectation value as a `Float64`.

  ### Notes

  This function utilizes `ITensors.inner` to perform the contraction $\langle \psi | \hat{O} | \psi \rangle$. It returns the real part of the result, which is appropriate for Hermitian operators representing physical observables.
  """
  return real(inner(ψ', O, ψ))
end

#------------ Operator Entropy ------------
function compute_entropy(s::Union{Float64, Vector{Float64}, NDTensors.DenseTensor{}})::Float64
  raw"""
  compute_entropy(s::Union{Float64, Vector{Float64}, NDTensors.DenseTensor{}})::Float64

  Calculate the Shannon entropy of the singular value spectrum of an MPO or MPS.

  ### Arguments

  * `s`: A vector of singular values (e.g., from the SVD of a tensor in the network) or one singular value.

  ### Returns

  * The Shannon entropy $S = -\sum p_i \ln(p_i)$ as a `Float64`, where $p_i$ are the normalized squared singular values.

  ### Notes

  The function normalizes the squared singular values to form a probability distribution and ignores values smaller than $10^{-18}$ to maintain numerical stability.
  """
  p = s.^2 / sum(s.^2)
  p = p[p .> 1e-18]
  return -sum(p .* log.(p))
end

function operator_entropy(O::MPO, bond::Int, k::Int=1)::Float64
  raw"""
  operator_entropy(O::MPO, bond::Int, k::Int=1)::Float64

  Calculate the entanglement entropy of an MPO at a specified bond by analyzing its singular value spectrum.

  ### Arguments

  * `O`: The `MPO` operator.
  * `bond`: The bond index at which to compute the entropy.
  * `k`: The order of the Rényi entropy (default is 1).

  ### Returns

  * The entropy value as a `Float64`.

  ### Notes

  This function orthogonalizes the MPO relative to the specified bond, performs a singular value decomposition (SVD) on the local tensor, and computes the entropy of the resulting singular values. If `bond` is 1, the function returns 0.0, as there is no entanglement link to the left.
  """
  if bond == 1 # on ne peut pas couper au bond 1 (pas de lien à gauche)
    return 0.0
  end
    
  O_ortho = orthogonalize(O, bond)
  
  l_link = linkind(O_ortho, bond-1)
  s_inds = siteinds(O_ortho, bond)
    
  # SVD, on sépare le lien gauche et les indices physiques
  U, S, V = svd(O_ortho[bond], l_link, s_inds...)
    
  s_vals = diag(S)

  if k == 1
    return compute_entropy(s_vals) #Shannon entropie
  end
  
  norm_sq = sum(abs(s_vals)^2) # compute \sum_\alpha |s_\alpha|²  
  if norm_sq == 0
    return 0.
  end

  sum_coeffs_2k = sum(abs(s_vals)^(2*k))
  zeta_k = sum_coeffs_2k / (norm_sq^k)

  return log(zeta_k) / (1-k)

end

#------------ Noise layer ------------
function apply_depolarizing_noise(
  O::MPO, 
  sites_to_noise::Vector{<:Index}, 
  lambda::Float64; 
  maxdim::Union{Nothing, Integer}=nothing, 
  cutoff::Float64=0.
  )::MPO
  raw"""
  apply_depolarizing_noise(O::MPO, sites_to_noise::Vector{<:Index}, lambda::Float64; maxdim::Union{Nothing, Integer}=nothing, cutoff::Float64=0.0)::MPO

  Apply a local depolarizing noise channel to specified sites of a Matrix Product Operator (MPO).

  ### Arguments

  * `O`: The input `MPO` to be modified.
  * `sites_to_noise`: A vector of `ITensors.Index` objects representing the sites where the noise is applied.
  * `lambda`: The strength of the depolarizing channel.
  * `maxdim`: The maximum bond dimension for truncation during MPO addition.
  * `cutoff`: The singular value truncation cutoff for MPO addition.

  ### Returns

  * A new `MPO` with the noise channel applied, compressed according to `maxdim` and `cutoff`.

  ### Notes

  The depolarizing channel is modeled as $\mathcal{E}(\rho) = (1 - \lambda)\rho + \frac{\lambda}{3}(X\rho X + Y\rho Y + Z\rho Z)$. This implementation applies the channel site-by-site, performing MPO summation and truncation at each step to manage bond dimension growth.
  """
  if maxdim === nothing
    maxdim = 2^length(O)
  end

  O_noisy = copy(O)
    
  c0 = 1.0 - (3.0 * lambda / 4.0)
  c1 = lambda / 4.0
    
  # On applique le bruit site par site
  for s in sites_to_noise
    X = op("X", s)
    Y = op("Y", s)
    Z = op("Z", s)
        
    # apply_dag=true fait X*O*X, Y*O*Y, Z*O*Z car Pauli est hermitien
    O_X = apply([X], O_noisy; apply_dag=true, cutoff=cutoff)
    O_Y = apply([Y], O_noisy; apply_dag=true, cutoff=cutoff)
    O_Z = apply([Z], O_noisy; apply_dag=true, cutoff=cutoff)
        
    # +(A, B; kwargs...) fait A+B avec truncation
    O_new = +(c0 * O_noisy, c1 * O_X; cutoff=cutoff, maxdim=maxdim)
    O_new = +(O_new, c1 * O_Y; cutoff=cutoff, maxdim=maxdim)
    O_new = +(O_new, c1 * O_Z; cutoff=cutoff, maxdim=maxdim)
        
    O_noisy = O_new
  end
  return O_noisy
end

#------------ Propagate Layer by layer ------------ 
function tensor_dag(U::ITensor)::ITensor
  raw"""
  tensor_dag(U::ITensor)::ITensor

  Compute the Hermitian conjugate of an `ITensor` representing an operator.

  ### Arguments

  * `U`: The input `ITensor` operator with a set of indices, where the first half are typically input indices and the second half are output indices.

  ### Returns

  * The adjoint `ITensor` with the site indices swapped to represent the conjugate operation.

  ### Notes

  This function performs a manual swap of site indices to effectively compute the dagger of an operator tensor. It assumes the tensor indices are structured such that input and output indices are paired in the first and second halves of the index collection.
  """
  indices = inds(U)
  N = length(indices) ÷ 2

  U_dag = U
  for i in 1:N 
    U_dag = swapind(U_dag, indices[i], indices[i+N])
  end
  return U_dag 
end

function propagate_1layer(
  layer::Union{ITensor, Vector{ITensor}}, 
  current::MPO,
  norm0::Float64;
  maxdim::Union{Nothing, Integer}=nothing, 
  cutoff::Float64=0.,
  γ::Float64=0., # for the Noise
  normalize::Bool=true
  )::MPO
  raw"""
  propagate_1layer(layer::Union{ITensor, Vector{ITensor}}, current::MPO, norm0::Float64; maxdim::Union{Nothing, Integer}=nothing, cutoff::Float64=0.0, γ::Float64=0.0)::MPO

  Propagate an MPO observable through a single layer of a quantum circuit, including optional depolarizing noise and renormalization.

  ### Arguments

  * `layer`: The gate(s) in the current layer, represented as `ITensor`(s).
  * `current`: The input `MPO` to propagate.
  * `norm0`: The target reference norm to maintain during renormalization.
  * `maxdim`: The maximum bond dimension for truncation.
  * `cutoff`: The singular value truncation threshold.
  * `γ`: The strength of the depolarizing noise channel.
  * `normalize` : If `true`, at each layer, it renormalizes the observable.

  ### Returns

  * The evolved `MPO` after gate application, noise injection, and renormalization.

  ### Notes

  This function evolves the operator according to the Heisenberg picture by applying the layer as $O \leftarrow U^\dagger O U$. If $\gamma > 0$, a local depolarizing noise channel is applied to all sites. Finally, the operator is rescaled by `norm0 / norm(current)` to compensate for any magnitude lost due to SVD truncations.
  """
  if maxdim === nothing
    maxdim = 2^length(current)
  end

  sites_mps = [siteind(current, i; plev=0) for i in 1:length(current)]

  current = apply(layer, current; apply_dag=true, cutoff=cutoff, maxdim=maxdim) # apply(U,0,apply_dag=true) fait U O U+ donc on applique d'abord le dag a layer pour avoir +U O U
  if !(γ == 0.)
    current = apply_depolarizing_noise(current, sites_mps, γ; cutoff, maxdim)
  end
  if normalize
    current *= (norm0/norm(current)) # pour conserver la norm malgres les troncations
  end
  return current
end

function propagate(
  circuit::Vector{Vector{ITensor}},
  observable::MPO;
  cutoff::Float64=0.,
  maxdim::Union{Int, Nothing}=nothing,
  bond::Union{Int, Nothing}=nothing, # for the Entropy
  k::Int64=1, # for the Entropy
  ψ0::Union{MPS, Nothing}=nothing, # for the Overlap
  γ::Float64=0., # for the Noise
  normalize::Bool=true,
  disable_print::Bool=false
  )::Tuple{MPO, Dict{String, Any}}
  raw"""
  propagate(circuit::Vector{Vector{ITensor}}, observable::MPO; cutoff::Float64=0.0, maxdim::Union{Int, Nothing}=nothing, bond::Union{Int, Nothing}=nothing, k::Int64=1, ψ0::Union{MPS, Nothing}=nothing, γ::Float64=0.0, disable_print::Bool=false)::Tuple{MPO, Dict{String, Any}}

  Propagate an MPO observable through a quantum circuit layer-by-layer in the Heisenberg picture.

  ### Arguments

  * `circuit`: A vector of layers, where each layer is a vector of `ITensor` gates.
  * `observable`: The initial `MPO` to propagate.
  * `cutoff`: Singular value truncation threshold for MPO applications.
  * `maxdim`: Maximum bond dimension allowed during MPO operations.
  * `bond`: Optional bond index to track entanglement entropy at each step.
  * `k`: The order of the Rényi entropy (default is 1).
  * `ψ0`: Optional reference `MPS` to track state overlap at each step.
  * `γ`: Intensity of the depolarizing noise channel.
  * `normalize` : If `true`, at each layer, it renormalizes the observable.
  * `disable_print`: If `true`, suppresses progress output.

  ### Returns

  * A `Tuple` containing:
  * `current`: The final propagated `MPO`.
  * `result`: A `Dict` containing tracked diagnostics: `"maxlink"`, `"norm"`, `"S"` (entropy), `"overlap"`, and total execution `"time"`.


  ### Notes

  This function processes the circuit in the Heisenberg picture, applying each layer as $U^\dagger O U$. It tracks bond dimensions, norms, and optionally entanglement and state overlaps throughout the evolution, ensuring renormalization at each step to maintain the initial norm.
  """
  t0 = time()
  nlayers = length(circuit)
  
  maxlink, entropies, norms, overlaps = Int[], Float64[], Float64[], Float64[]

  if maxdim === nothing
    maxdim = length(observable)
  end

  current = copy(observable)
  heinseberg_circuit = [reverse(tensor_dag.(layer)) for layer in reverse(circuit)]
  
  if bond != nothing
    push!(entropies, operator_entropy(observable, bond))
  end
  if ψ0 != nothing
    push!(overlaps, overlap(observable, ψ0))
  end
  
  norm0 = norm(observable)
  push!(norms, norm0)
  push!(maxlink, maxlinkdim(observable))

  for (layer_idx, layer) in enumerate(heinseberg_circuit)
    current = propagate_1layer(layer, current, norm0; maxdim, cutoff, γ, normalize)

    if bond != nothing
      push!(entropies, operator_entropy(current, bond))
    end
    if ψ0 != nothing
      push!(overlaps, overlap(current, ψ0))
    end
    push!(norms, norm(current))
    push!(maxlink, maxlinkdim(current))

    if layer_idx % max(1, nlayers ÷ 10)==0 && !disable_print
      println("layer : $layer_idx /$nlayers complete")
    end
  end

  elapsed_time = time() - t0
  println("Time taken by mpo_functions.propagate: ", elapsed_time, " seconds")

  result = Dict("maxlink" => maxlink, "norm" => norms, "S" => entropies, "overlap" => overlaps, "time"=> elapsed_time)

  return current, result
end

#------------ Find optimal truncations ------------ 
function find_truncations(
  tolerance::Float64, 
  circuit::Vector{Vector{ITensor}}, 
  observable::MPO;
  γ::Float64=0., # for the Noise
  disable_print::Bool=false
  )::Tuple{Int64, Float64}
  raw"""
  find_truncations(tolerance::Float64, circuit::Vector{Vector{ITensor}}, observable::MPO; γ::Float64=0.0, disable_print::Bool=false)::Tuple{Int64, Float64}

  Determine the optimal truncation parameters (`maxdim` and `cutoff`) for Matrix Product Operator (MPO) propagation.

  ### Arguments

  * `tolerance`: The relative tolerance used to check for convergence of the overlap sequence.
  * `circuit`: The sequence of circuit layers, where each layer is a vector of `ITensor` gates.
  * `observable`: The initial `MPO` to propagate.
  * `γ`: Intensity of the depolarizing noise channel.
  * `disable_print`: If `true`, suppresses progress output.

  ### Returns

  * A `Tuple` containing:
  * `maxdim`: The determined optimal maximum bond dimension.
  * `cutoff`: The determined optimal singular value truncation threshold.


  ### Notes

  This function performs a sensitivity analysis by iteratively propagating the MPO through the circuit. It systematically increases `maxdim` and then decreases `cutoff` until the relative change in the overlap with a reference $|0\dots0\rangle$ state falls within the specified `tolerance`. This helps find the minimum computational resources required to maintain physical accuracy.
  """
  N = length(observable)
  dim = 2^N
  nlayers = length(circuit)
  norm0 = norm(observable)

  sites_mps = [siteind(observable, i; plev=0) for i in 1:N]
  init_state = ["Up" for _ in 1:N] # "Dn" pour down
  ψ0 = MPS(sites_mps, init_state)

  heinseberg_circuit = [reverse(tensor_dag.(layer)) for layer in reverse(circuit)]

  smaller_cutoff = 1e-13
  
  # ----- Max Dim TEST -----
  maxdim = 3
  
  overlaps = fill(Inf, nlayers)
  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    maxdim += max(1, dim÷10)

    if maxdim >= dim
      maxdim = dim
      break
    end

    current = copy(observable)
    overlaps = Float64[]
    for layer in heinseberg_circuit
      current = propagate_1layer(layer, current, norm0; maxdim, cutoff=smaller_cutoff, γ)
      push!(overlaps, overlap(current, ψ0))
    end

    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end

  # ----- Cutoff TEST -----
  cutoff_power = -1

  overlaps = fill(Inf, nlayers)
  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    cutoff_power -= 1
    cutoff = 10^float(cutoff_power)

    if cutoff <= smaller_cutoff
      if !disable_print
        println("Optimal truncations find : (Maxdim=$maxdim, Cutoff=$smaller_cutoff)")
      end
      return maxdim, smaller_cutoff
    end
    
    current = copy(observable)
    overlaps = Float64[]
    for layer in heinseberg_circuit
      current = propagate_1layer(layer, current, norm0; maxdim, cutoff, γ)
      push!(overlaps, overlap(current, ψ0))
    end

    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end
  cutoff = 10^float(cutoff_power)
  if !disable_print
    println("Optimal truncations find : (Maxdim=$maxdim, Cutoff=1e$cutoff_power)")
  end
  return maxdim, cutoff
end

end # module