# Provider config variants for m-aia

How each campaign config maps onto the m-aia solver + CMI providers.

Provider selection is by env `CPP_ML_INTERFACE_PROVIDER_ENV` (`AIX` default,
`PHYDLL`, `SMARTSIM`), device by `CPP_ML_INTERFACE_DEVICE=CPU`. The solver picks
`./config_aix.toml`, `./config_phydll.toml`, or `./config_smartsim.toml`
accordingly and applies runtime overrides from `properties.toml`.

## AIX

- Config: `config_aix.toml` + `communication_mode = "collective"` (set by the
  runner).
- Solver overrides: `library.app_comm` (opaque `MPI_Comm` handle),
  `library.model_file`, `library.batchsize` (default 50000, `MAIA_AIX_BATCHSIZE`).
- No oversubscription: one solver rank per core.

## PhyDLL C++

Two variants, differing only in `library.solver_readiness_wait`:

| Variant | `config_phydll.toml` | Extra procs |
|---|---|---|
| `phydll-blocking` | key omitted (defaults false) | 1 DL rank per solver rank |
| `phydll-readiness` | `solver_readiness_wait = true` | 1 DL rank per solver rank |

- Solver overrides: `library.model_file`, `library.backend=TORCH`,
  `library.device`, `library.batch_size` (0 on CPU).
- MPMD: `2 x ranks` processes on `ranks` cores (oversubscribed), e.g. 96 solver +
  96 DL. `PHYDLL_MPMD_SHUTDOWN_BARRIER=1` and `PHYDLL_DL_FIELD_COUNT=2`.

## SmartSim

The solver's SmartSim branch overrides only `library.device`,
`library.model_backend`, `library.model_path`, `library.model_name` (and
`num_gpus` on GPU). Therefore:

- **Threads / binding** come from the controller CLI
  (`CPP-ML-Interface/dl_clients/smartsim_controller.py`):
  `--threads-per-queue`, `--intra-op-threads`, `--inter-op-threads`,
  `--cpu-cores-per-node`, or `--use-default-cpu-settings`.
- **DB batching** comes from `config_smartsim.toml` `[library]`:
  `batch_size`, `min_batch_size`, `min_batch_timeout` (all 0 for TS parity).

TS-parity variants (`mini_app/campaign_cpu_terrain.sbatch`):

| Config | `--use-default-cpu-settings` | `--threads-per-queue` | `--intra-op-threads` | `--cpu-cores-per-node` | `batch_size` |
|---|---|---|---|---|---|
| `SS_DEFAULT` | yes (omit all) | – | – | – | 0 |
| `SS_TPQ1_I96_B96` | no | 1 | 96 | 96 | 0 |
| `SS_TPQ12_I8_B96` | no | 12 | 8 | 96 | 0 |
| `SS_TPQ96_I1_B96` | no | 96 | 1 | 96 | 0 |

The controller is launched locally (loopback, auto port) with spare allocated
cores; DB processes are not placed in a separate Slurm step or affinity mask
(functional parity with the TS smoke, not a controlled-placement benchmark).
