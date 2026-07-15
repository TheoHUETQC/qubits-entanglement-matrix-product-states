module exact_functions
export overlap, apply_depolarizing_noise, propagate_layerbylayer, get_Zi

using LinearAlgebra

#------------ matrix 2x2 -> matrix 2^Nx2^N ------------
function local_to_global_matrices(U::Matrix, idx::Int64, nqubits::Int64)::Matrix{ComplexF64}
  raw"""
  local_to_global_matrices(U::Matrix, idx::Int64, nqubits::Int64)::Matrix{ComplexF64}

  Embeds a local qubit operator `U` into the global Hilbert space of a system containing `nqubits`.

  # Arguments
  - `U::Matrix{ComplexF64}`: The local operator (matrix) acting on the qubit pair.
  - `idx::Int64`: A integer indicating the target qubit indices. 
  - `nqubits::Int64`: The total number of qubits in the system.

  # Returns
  - `Matrix{ComplexF64}`: The resulting global operator matrix in the full Hilbert space.
  """

  U_global = [1.0 + 0.0im;;]
  
  for k in 1:nqubits
      if k == idx
          U_global = kron(U_global, U)
      else
          U_global = kron(U_global, I(2))
      end
  end
  return U_global
end

#------------ Overlap with a state psi ------------
function overlap(O::Matrix, ψ::Union{Vector{Float64}, Vector{Int64}})::Float64
    raw"""
    overlap(O::Matrix, ψ::Union{Vector{Float64}, Vector{Int64}})::Float64

    Compute the expectation value $\langle \psi | \hat{O} | \psi \rangle$ of a dense matrix observable for a given state vector.

    ### Arguments

    * `O`: The matrix representation of the operator.
    * `ψ`: The state vector $\psi$.

    ### Returns

    * The real-valued expectation value as a `Float64`.

    ### Notes

    This function performs the matrix-vector multiplication $\langle \psi | \hat{O} | \psi \rangle$. It returns the real part of the result, which is appropriate for physical observables represented by Hermitian matrices.
    """
    return real(ψ' * O * ψ)
end

#------------ Noise layer ------------
function apply_depolarizing_noise(
  O::Matrix,
  lambda::Float64
  )::Matrix
  raw"""
  apply_depolarizing_noise(O::Matrix, lambda::Float64)::Matrix

  Apply a local depolarizing noise channel.

  ### Arguments

  * `O`: The input `Matrix` to be modified.
  * `lambda`: The strength of the depolarizing channel.

  ### Returns

  * A new `Matrix` with the noise channel applied.

  ### Notes

  The depolarizing channel is modeled as $\mathcal{E}(\rho) = (1 - \lambda)\rho + \frac{\lambda}{3}(X\rho X + Y\rho Y + Z\rho Z)$.
  """

    dim = size(O, 1)
    nqubits = Int(log2(dim))
    
    # Pauli Matrices
    X_base = ComplexF64[0 1; 1 0]
    Y_base = ComplexF64[0 -im; im 0]
    Z_base = ComplexF64[1 0; 0 -1]
    
    c0 = 1.0 - (3.0 * lambda / 4.0)
    c1 = lambda / 4.0
    
    O_noisy = copy(O)

    for target_qubit in 1:nqubits        
        X_global = local_to_global_matrices(X_base, target_qubit, nqubits)
        Y_global = local_to_global_matrices(Y_base, target_qubit, nqubits)
        Z_global = local_to_global_matrices(Z_base, target_qubit, nqubits)
        
        O_noisy = c0 * O_noisy + c1 * ((X_global * O_noisy * X_global) + (Y_global * O_noisy * Y_global) + (Z_global * O_noisy * Z_global))
    end
    
    return O_noisy
end

#------------ Propagate Layer by layer ------------ 
function propagate(    
    circuit::Vector{Vector{Matrix}},
    observable::Matrix;
    bond::Union{Int, Nothing}=nothing,
    ψ0::Union{Vector{Float64}, Nothing}=nothing,
    γ::Float64=0., # for the Noise
    normalize::Bool=true,
    disable_print::Bool=false
    )::Tuple{Matrix, Dict{String, Any}}
    raw"""
    propagate(circuit::Vector{Vector{Matrix}}, observable::Matrix; bond::Union{Int, Nothing}=nothing, ψ0::Union{Vector{Float64}, Nothing}=nothing, disable_print::Bool=false)::Tuple{Matrix, Dict{String, Any}}

    Propagate a dense matrix observable through a quantum circuit layer-by-layer in the Heisenberg picture.

    ### Arguments

    * `circuit`: A vector of layers, where each layer is a vector of unitary `Matrix` gates.
    * `observable`: The initial dense `Matrix` to propagate.
    * `bond`: Optional bond index to track entanglement entropy at each step.
    * `ψ0`: Optional reference state vector to track state overlap at each step.
    * `γ`: Intensity of the depolarizing noise channel.
    * `normalize` : If `true`, at each layer, it renormalizes the observable.
    * `disable_print`: If `true`, suppresses progress output.

    ### Returns

    * A `Tuple` containing:
    * `current`: The final propagated `Matrix`.
    * `result`: A `Dict` containing tracked diagnostics: `"norm"`, `"S"` (entropy), `"overlap"`, and total execution `"time"`.


    ### Notes

    This function evolves the operator according to the Heisenberg picture ($O \leftarrow U^\dagger O U$) by iterating through the circuit in reverse order. It computes exact matrix products and tracks physical diagnostics at each layer if requested.
    """
    t = time()
    nlayers = length(circuit)

    entropies, norms, overlaps = Float64[], Float64[], Float64[]

    if bond != nothing
      push!(entropies, operator_entropy(observable, bond))
    end
    if ψ0 != nothing
      push!(overlaps, overlap(observable, ψ0))
    end
    norm0 = norm(observable)
    push!(norms, norm0)

    current = copy(observable)
    for (layer_idx, layer) in enumerate(reverse(circuit))
        for gate in reverse(layer)
          current = gate' * current * gate
        end
        if !(γ == 0.)
          current = apply_depolarizing_noise(current, γ)
          if normalize
            current *= (norm0/norm(current))
          end
        end
        if bond != nothing
          push!(entropies, operator_entropy(current, bond))
        end
        if ψ0 != nothing
          push!(overlaps, overlap(current, ψ0))
        end
        push!(norms, norm(current))

        if layer_idx % max(1, nlayers ÷ 10)==0 && !disable_print
            println("layer : $layer_idx /$nlayers complete")
        end
    end

    elapsed_time = time() - t
    println("Time taken by exact_functions.propagate: ", elapsed_time, " seconds")

    result = Dict("norm" => norms, "S" => entropies, "overlap" => overlaps, "time" => elapsed_time)

    return current, result
end

#------------ Observable for test ------------ 

function get_Zi(nqubits::Int64, target_qubit_idx::Int64)
    raw"""
    get_Zi(nqubits::Int64, target_qubit_idx::Int64)::Matrix{ComplexF64}

    Construct the dense matrix representation of a single-qubit Pauli-Z operator acting on the $i$-th qubit (target_qubit_idx) of an $n$-qubit system.

    ### Arguments

    * `nqubits`: The total number of qubits in the system.
    * `target_qubit_idx`: The index of the qubit where the Z operator is applied (1-indexed).

    ### Returns

    * A dense `Matrix{ComplexF64}` of size $2^n \times 2^n$ representing the operator $I \otimes \dots \otimes Z_i \otimes \dots \otimes I$.
    """
    
    Z_base = ComplexF64[1 0; 0 -1]

    Z_global = local_to_global_matrices(Z_base, target_qubit_idx, nqubits)

    return Z_global
end

end # module