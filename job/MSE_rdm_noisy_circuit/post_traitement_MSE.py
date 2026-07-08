from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# for the colormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import ast
def convert_to_list(data_series):
  return ast.literal_eval(data_series)

# Define the directory to save figures
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True) # Create the directory if it doesn't exist

# --- Paramètres ---
nqubits_list = [7]
lambda_list = np.arange(0, 0.4, 0.03)
runs = range(50)

def compute_maxdim_list(nqubits):
  power_list = np.array(range(nqubits-5, nqubits))
  return 2 ** (power_list)

def compute_maxsize_list(nqubits):
  power_list = np.array(range(nqubits-5, nqubits))
  return 4 ** (power_list)

def compute_thermalisation(nlayers):
  return -1

def get_data(run, nqubits, method_type, col_name):
  """Récupère la colonne spécifique du CSV
  method_type = "pp" or "mpo"
  """

  path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_MSE/test/run_{run}/results_N_{nqubits}-MSE_{method_type}.csv"
  df = pd.read_csv(path)
  return df[col_name]

# --- Calcul et Plot ---
for nqubits in nqubits_list:
    nlayers = 2*nqubits
    thermalisation = compute_thermalisation(nlayers)

    maxdim_list = compute_maxdim_list(nqubits)
    maxsize_list = compute_maxsize_list(nqubits)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=lambda_list.min(), vmax=lambda_list.max())
    for lambda_val in lambda_list:
      lambda_val = round(lambda_val, 2)

      pp_mse_vs_truncation_means, mpo_mse_vs_truncation_means = [], []
      pp_mse_vs_truncation_errs, mpo_mse_vs_truncation_errs  = [], []

      for truncation_idx in range(len(maxsize_list)):
        # recupere la data sur les runs
        all_run_pp_mse = []
        all_run_mpo_mse = []
        for r in runs:
          exact_value_sq = np.mean(get_data(r+1, nqubits, "pp", f"exact value sq, gammaN={lambda_val}")[thermalisation:])

          col_name_mpo = f"sq error, gammaN={lambda_val}, maxdim={maxdim_list[truncation_idx]}"
          col_name_pp = f"sq error, gammaN={lambda_val}, maxsize={maxsize_list[truncation_idx]}"
          pp_mse = np.mean(get_data(r+1, nqubits, "pp", col_name_pp)[thermalisation:])
          mpo_mse = np.mean(get_data(r+1, nqubits, "mpo", col_name_mpo)[thermalisation:])

          all_run_pp_mse.append(pp_mse)
          all_run_mpo_mse.append(mpo_mse)

        # Calcul stats sur les runs
        pp_mse_means = (np.mean(all_run_pp_mse))
        pp_mse_errs = (np.std(all_run_pp_mse) / np.sqrt(len(runs)))

        mpo_mse_means = (np.mean(all_run_mpo_mse))
        mpo_mse_errs = (np.std(all_run_mpo_mse) / np.sqrt(len(runs)))

        pp_mse_vs_truncation_means.append(pp_mse_means)
        pp_mse_vs_truncation_errs.append(pp_mse_errs)
        mpo_mse_vs_truncation_means.append(mpo_mse_means)
        mpo_mse_vs_truncation_errs.append(mpo_mse_errs)
      # Plot
      color = cmap(norm(lambda_val))
      if lambda_val == 0.21:
        color = "red"
      axes[0].errorbar(np.array(maxsize_list)/(4**nqubits), pp_mse_vs_truncation_means, yerr=pp_mse_vs_truncation_errs, marker='o', label=f"λ={lambda_val:.2f}", capsize=3, color=color)
      axes[1].errorbar(np.array(maxdim_list)/(2**nqubits), mpo_mse_vs_truncation_means, yerr=mpo_mse_vs_truncation_errs, marker='s', capsize=3, color=color)

    axes[0].set_title(f"nqubits={nqubits},"+ r'MSE $Tr[\rho(O-O_{truncate})]^2$ (Pauli)')
    axes[1].set_title(r'MSE $Tr[\rho(O-O_{truncate})]^2$ (MPO)')
    axes[0].set_xlabel(r'Max size $N_p / 4^N$')
    axes[1].set_xlabel(r'max bond dim $\chi / 2^N$')
    for ax in axes:
        ax.set_ylabel("MSE")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xscale('log')
        #ax.set_yscale('log')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("Gamma * N", rotation=270, labelpad=15)

    plt.show()

    fig_name = f"overlap_MSE-nqubits_{nqubits}.png"
    fig_path = os.path.join(save_dir, fig_name)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)