"""Local contract tests. Never call real sbatch/srun or artifact executables."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify", HERE / "verify_cpu_run.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class RetestTests(unittest.TestCase):
    def test_submission_matrices(self):
        for mode, count, mm in [("profile", 21, 10), ("memory", 21, 10), ("timing", 21, 10)]:
            env = dict(os.environ, MAIA_BUILD_DIR="/mock/build")
            # Exercise real submission control flow, but replace sbatch with a function.
            command = 'sbatch() { printf "MOCK %s %s %s %s\\n" "$CASE_NAME" "$PROVIDER" "${PHYDLL_MODE:-${SS_CONFIG:-AIX}}" "$MAIA_BUILD_DIR"; }; export -f sbatch; bash "$1" "$2" 123 --dry-run'
            result = subprocess.run(["bash", "-c", command, "test", str(HERE / "slurm/submit_cpu_retest.sh"), mode], env=env, text=True, capture_output=True, check=True)
            cells = [line for line in result.stdout.splitlines() if line.startswith("CELL ")]
            self.assertEqual(len(cells), count)
            self.assertEqual(sum("partition=c23mm" in line for line in cells), mm)
            self.assertEqual(len(set(line.split()[1] for line in cells)), count)
            self.assertEqual(sum("mlp_ov4" in line and "overlap=4" in line for line in cells), 7)
            self.assertNotIn("mlp_ov6", result.stdout)
            self.assertNotIn("partition=c23ml", result.stdout)
            self.assertIn("afterok:123", result.stdout)
            mocked = subprocess.run(["bash", "-c", command.replace("123 --dry-run", "123"), "test", str(HERE / "slurm/submit_cpu_retest.sh"), mode], env=env, text=True, capture_output=True, check=True)
            self.assertEqual(mocked.stdout.count("MOCK "), count)

    def test_filtered_submission(self):
        env = dict(os.environ, MAIA_BUILD_DIR="/mock/build", RETEST_FILTER="mlp_ov4_", RETEST_EXCLUDE_NODES="n23m0348,n23m0011")
        result = subprocess.run(["bash", str(HERE / "slurm/submit_cpu_retest.sh"), "timing", "123", "--dry-run"], env=env, text=True, capture_output=True, check=True)
        self.assertEqual(sum(line.startswith("CELL ") for line in result.stdout.splitlines()), 7)
        self.assertIn("JOBS=7", result.stdout)
        self.assertEqual(result.stdout.count("--exclude=n23m0348\\,n23m0011"), 7)

    def test_completed_build_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            env = dict(os.environ, MAIA_BUILD_DIR=str(build), RETEST_FILTER="mlp_ov4_AIX_")
            command = ["bash", str(HERE / "slurm/submit_cpu_retest.sh"), "timing", "0", "--dry-run"]
            rejected = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 2)
            (build / "BUILD_READY").touch()
            (build / "bin").mkdir()
            binary = build / "bin/maia"
            binary.touch()
            binary.chmod(0o755)
            accepted = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            self.assertIn("JOBS=1", accepted.stdout)
            self.assertNotIn("--dependency", accepted.stdout)

    def test_mocked_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            build = root / "build"
            commands = root / "commands"
            for directory in [artifact / "input", artifact / "slurm", artifact / "CPP-ML-Interface/dl_clients", build / "bin", commands]:
                directory.mkdir(parents=True)
            (artifact / "setup_env_claix23.sh").write_text(":\n")
            for name in ["grid_les_medium.hdf5", "restart_les_init_medium.hdf5", "model.pt"]:
                (artifact / "input" / name).touch()
            properties = {"timeSteps": 300, "hostFraction": '"0.2"', "mlInterval": 1, "mlInputLength": 5,
                          "mlStepCoefficient": 25, "mlForecastWindow": 2, "mlScalingFactor": 1,
                          "mlInputStepDistance": 1, "mlCubeOverlap": 0, "solutionInterval": 50, "modelPath": '"bad"'}
            (artifact / "input/properties_run_les_ref_medium.toml").write_text("\n".join(f"{k} = {v}" for k, v in properties.items()) + "\n")
            for provider in ["aix", "phydll", "smartsim"]:
                (artifact / f"config_{provider}.toml").write_text("[library]\nbatch_size = 123\ncommand_timeout = 600\nsocket_timeout = 600\nmodel_timeout = 600000\n")
            (build / "CMakeCache.txt").touch()
            for binary in [build / "bin/maia", build / "bin/phydll_dl_client"]:
                binary.write_text("#!/bin/bash\nexit 99\n")
                binary.chmod(0o755)
            srun = commands / "srun"
            srun.write_text(f"#!{sys.executable}\n" + '''import io, os, pathlib, tarfile
args = __import__('sys').argv[1:]
pathlib.Path('launch.args').write_text(' '.join(args))
if os.environ['PROVIDER'] == 'SMARTSIM':
    assert os.environ['SR_CMD_TIMEOUT'] == '3600'
    assert os.environ['SR_SOCKET_TIMEOUT'] == '3600000'
    assert os.environ['SR_MODEL_TIMEOUT'] == '3600000'
count = int(os.environ['EXPECTED_INFERENCE_CALLS'])
for i in range(count):
    print(f'ML_STEP_TIMING step={15 + 29*i} ml_ms=1 rank_min_ms=1 rank_max_ms=1')
if os.environ['MEASUREMENT'] == 'profile':
    n = 192 if os.environ['PROVIDER'] == 'PHYDLL' else 96
    regions = ['m-AIA-Solver::run', 'dl_recv', 'dl_send', 'dl_torch_forward']
    xml = '<cube><system>' + ''.join(f'<locationgroup><type>process</type><rank>{i}</rank></locationgroup>' for i in range(n)) + '</system><program>' + ''.join(f'<region><name>{r}</name></region>' for r in regions) + '</program></cube>'
    dest = pathlib.Path(os.environ['SCOREP_EXPERIMENT_DIRECTORY'])
    dest.mkdir()
    with tarfile.open(dest / 'profile.cubex', 'w') as archive:
        member = tarfile.TarInfo('anchor.xml'); member.size = len(xml)
        archive.addfile(member, io.BytesIO(xml.encode()))
''')
            srun.chmod(0o755)
            activate = root / "activate"
            activate.write_text(":\n")
            (artifact / "CPP-ML-Interface/dl_clients/smartsim_controller.py").write_text("from pathlib import Path\nimport time\nPath('.ssdb_endpoint').write_text('localhost:1234\\n')\nwhile not Path('.solver_done').exists(): time.sleep(.01)\n")
            env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}", PROJECT_DIR=str(HERE.parent),
                       ARTIFACT_DIR=str(artifact), MAIA_BUILD_DIR=str(build), SCRATCH_ROOT=str(root / "runs"),
                       SLURM_JOB_ID="123", SOLVER_RANKS="96", NP_DL="96", MODEL=str(artifact / "input/model.pt"),
                       SMARTSIM_PYTHON_ENV=str(activate), READY_TIMEOUT_SECONDS="5", CASE_TIMEOUT_SECONDS="10")
            for mode in ["timing", "profile", "memory"]:
                for provider in ["AIX", "PHYDLL", "SMARTSIM"]:
                    settings = dict(env, PROVIDER=provider, MEASUREMENT=mode, EXPECTED_INFERENCE_CALLS="3" if mode == "memory" else "7", RUN_STEPS="80" if mode == "memory" else "200")
                    command = 'nm() { if [[ "$MEASUREMENT" == profile ]]; then printf "SCOREP_InitMeasurement\\n"; else printf "main\\n"; fi; }; export -f nm; bash "$1"'
                    result = subprocess.run(["bash", "-c", command, "test", str(HERE / "slurm/cpu_retest.sbatch")], env=settings, capture_output=True, text=True)
                    # Local hosts have no Slurm job cgroup: memory must fail closed.
                    self.assertEqual(result.returncode, 1 if mode == "memory" else 0, result.stdout + result.stderr)
                    matching_runs = []
                    for candidate in (root / "runs").iterdir():
                        metadata = dict(line.split("=", 1) for line in (candidate / "run.info").read_text().splitlines() if "=" in line)
                        if metadata.get("PROVIDER") == provider and metadata.get("MEASUREMENT") == mode:
                            matching_runs.append(candidate)
                    self.assertEqual(len(matching_runs), 1)
                    run = matching_runs[0]
                    if provider == "SMARTSIM":
                        config = (run / "config_smartsim.toml").read_text()
                        self.assertIn("command_timeout = 3600", config)
                        self.assertIn("socket_timeout = 3600000", config)
                        self.assertIn("model_timeout = 3600000", config)
                        self.assertIn("SS_MODEL_TIMEOUT_MS=3600000", (run / "run.info").read_text())
                        self.assertIn("SS_SOCKET_TIMEOUT_MS=3600000", (run / "run.info").read_text())
                    if mode == "memory":
                        self.assertIn("MEMORY_VALID=false", (run / "run.info").read_text())
                    else:
                        self.assertNotIn("MEMORY_PEAK_BYTES", (run / "run.info").read_text())
                    launch = (run / "launch.args").read_text()
                    self.assertIn(f"--ntasks={192 if provider == 'PHYDLL' else 96}", launch)
                    if provider == "PHYDLL":
                        self.assertIn("0-95", (run / "phydll.conf").read_text())
                        self.assertIn("96-191", (run / "phydll.conf").read_text())

    def test_profile_topology_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run.info").write_text("EXPECTED_INFERENCE_CALLS=3\nMEASUREMENT=profile\nSOLVER_RANKS=96\nPROVIDER=AIX\n")
            (run / "solver.log").write_text("\n".join(f"ML_STEP_TIMING step={s}" for s in [15, 44, 73]))
            (run / "scorep").mkdir()
            xml = b"<cube><locationgroup><type>process</type><rank>0</rank></locationgroup></cube>"
            with tarfile.open(run / "scorep/profile.cubex", "w") as archive:
                member = tarfile.TarInfo("anchor.xml")
                member.size = len(xml)
                archive.addfile(member, io.BytesIO(xml))
            with self.assertRaisesRegex(ValueError, "MPI ranks"):
                verify.verify(run)

    def test_devel_diagnostic_plan(self):
        script = HERE / "slurm/debug_transformer_smartsim.sbatch"
        for config, counts in [("tpq96", [3, 7]), ("default", [1])]:
            result = subprocess.run(["bash", str(script), config, "--plan"], text=True, capture_output=True, check=True)
            rows = [dict(field.split("=", 1) for field in line.split()) for line in result.stdout.splitlines()]
            self.assertEqual([int(row["calls"]) for row in rows], counts)
            self.assertTrue(all(row["socket_ms"] == "3600000" for row in rows))
            # Reserve controller readiness/cleanup overhead within the 1-hour allocation.
            self.assertLess(sum(int(row["case_s"]) + 240 for row in rows), 3600)


if __name__ == "__main__":
    unittest.main()
