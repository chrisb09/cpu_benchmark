#!/bin/bash
#SBATCH --job-name=PHYDLL_CPP_DLR96_I1
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --partition=c23mm
#SBATCH --exclusive
#SBATCH --account=thes2181
#SBATCH --mem=238G
#SBATCH --output=/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23mm/logs/PHYDLL_CPP_DLR96_I1_%j.log

set -euo pipefail
export PNAME="PHYDLL_CPP_DLR96_I1"
export CLIENT_KIND="cpp"
export RESULTS_CSV="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23mm/PHYDLL_CPP_DLR96_I1.csv"
exec /hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23mm/scripts/run_phydll_c23mm.sh
