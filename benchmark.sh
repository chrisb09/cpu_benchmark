sbatch driver.sh --clean
sbatch --partition=c23ms --account=thes2181 --export=ALL,BENCH_MODEL="transformer",BENCH_INPUTS=1000000 driver.sh --clean
sbatch --partition=c23ms --account=thes2181 --export=ALL,BENCH_MODEL="watercnn",BENCH_INPUTS=10000000 driver.sh --clean
sbatch --partition=c23ms --account=thes2181 --export=ALL,BENCH_MODEL="perfect",BENCH_INPUTS=100000000 driver.sh --clean
sbatch --partition=c23ms --account=thes2181 --export=ALL,BENCH_MODEL="mmcp_transformer",BENCH_INPUTS=10000,BENCH_SCHEMA="mmcp" driver.sh --clean
