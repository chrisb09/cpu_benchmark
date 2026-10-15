# m-aia Second-Call Check

## Scope

Read-only analysis of all 21 selected final plain-timing runs (147 ML calls),
including MLP ov0 as a supplement and both PhyDLL modes. Job selection comes
from `regenerate_cpu_final.py:JOBS`. `check_maia_warmup.py` reparses the original
logs and validates metadata and the seven-call schedule using the existing
strict selector. No policy, raw data, metadata, or figures were changed.

Calls 1-7 occur at solver steps 15, 44, 73, 102, 131, 160, 189. Durations are
synchronized solver-rank maxima from plain runs, not Score-P runs.

## Results

The reference is the arithmetic mean of calls 3-7. "Mean change" is the
relative change in the arithmetic mean if call 2 is added, i.e. using calls
2-7 instead. It equals one sixth of "Call 2 vs mean", before rounding.

| Workload | Configuration | Job | Call 2 (s) | Mean 3-7 (s) | Call 2 vs mean | Mean change |
|---|---|---:|---:|---:|---:|---:|
| Transformer | AIX | 4952696 | 28.905 | 21.084 | +37.09% | +6.18% |
| Transformer | PhyDLL blocking | 4940320 | 72.016 | 57.571 | +25.09% | +4.18% |
| Transformer | PhyDLL readiness | 4940323 | 26.960 | 21.744 | +23.99% | +4.00% |
| Transformer | SS_DEFAULT | 4966829 | 1456.106 | 1462.679 | -0.45% | -0.07% |
| Transformer | SS_TPQ1_I96_B96 | 4966830 | 1019.535 | 1034.821 | -1.48% | -0.25% |
| Transformer | SS_TPQ12_I8_B96 | 4966831 | 178.130 | 199.942 | -10.91% | -1.82% |
| Transformer | SS_TPQ96_I1_B96 | 4966832 | 31.663 | 27.910 | +13.45% | +2.24% |
| MLP ov0 | AIX | 4952712 | 0.084 | 0.089 | -5.64% | -0.94% |
| MLP ov0 | PhyDLL blocking | 4940326 | 0.475 | 0.450 | +5.61% | +0.94% |
| MLP ov0 | PhyDLL readiness | 4940329 | 0.445 | 0.422 | +5.44% | +0.91% |
| MLP ov0 | SS_DEFAULT | 4952716 | 1.788 | 1.501 | +19.09% | +3.18% |
| MLP ov0 | SS_TPQ1_I96_B96 | 4952718 | 16.856 | 16.456 | +2.43% | +0.41% |
| MLP ov0 | SS_TPQ12_I8_B96 | 4952719 | 1.061 | 0.589 | +80.13% | +13.36% |
| MLP ov0 | SS_TPQ96_I1_B96 | 4952721 | 1.238 | 0.824 | +50.15% | +8.36% |
| MLP ov4 | AIX | 4952723 | 0.262 | 0.272 | -3.64% | -0.61% |
| MLP ov4 | PhyDLL blocking | 4952726 | 4.220 | 3.974 | +6.18% | +1.03% |
| MLP ov4 | PhyDLL readiness | 4952727 | 3.598 | 3.423 | +5.10% | +0.85% |
| MLP ov4 | SS_DEFAULT | 4952728 | 9.002 | 9.401 | -4.24% | -0.71% |
| MLP ov4 | SS_TPQ1_I96_B96 | 4952729 | 12.448 | 12.794 | -2.70% | -0.45% |
| MLP ov4 | SS_TPQ12_I8_B96 | 4952730 | 6.190 | 6.706 | -7.69% | -1.28% |
| MLP ov4 | SS_TPQ96_I1_B96 | 4952731 | 7.421 | 6.593 | +12.56% | +2.09% |

## Interpretation

- Transformer AIX and both PhyDLL modes show two elevated initial calls,
  followed by five tightly clustered calls. The later ranges are respectively
  20.952-21.153 s, 57.444-57.737 s, and 21.697-21.816 s. Call 2 is clearly
  outside those observed ranges, consistent with a remaining startup effect.
- Transformer SS_TPQ96 also has an elevated second call (31.663 s versus a
  later range of 27.585-28.110 s). SS_DEFAULT and SS_TPQ1 have little second-call
  difference. SS_TPQ12 instead has a faster second call and variable later
  durations, not the same clean startup pattern.
- MLP ov4 does not show the Transformer AIX/PhyDLL pattern. Adding call 2 changes
  their means by approximately -0.61%, +1.03%, and +0.85%. SS_TPQ96's second
  call is above all five later calls, but later calls themselves vary.
- Supplementary MLP ov0 has pronounced second-call excesses in SS_TPQ12 and
  SS_TPQ96. For SS_TPQ12, call 2 is 1.061 s while all later calls are
  0.550-0.619 s. SS_TPQ96 has broader later variation (0.568-1.054 s), so
  excluding two calls does not by itself establish stationarity there.

These observations support retaining two excluded calls as a common m-aia
policy rather than declaring call 2 representative everywhere. They do not
establish a causal mechanism, an optimal warmup count, or that every subsequent
call is steady. Each configuration has only one independent plain run; the
five later calls are correlated sequential observations. No hypothesis tests
or independent-run confidence intervals are assigned to this comparison.

The timing evidence supports a profiling policy, but does not directly measure
startup effects in the instrumented runs. Existing AIX/PhyDLL phase aggregates
still have their previously documented seven/six-call scopes. No profiling
reruns were submitted as part of this check.

## Reproduce

From the CPU benchmark workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 mmcp_benchmark/check_maia_warmup.py
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 -m unittest discover -s mmcp_benchmark -p test_check_maia_warmup.py -v
```

The script also prints the calls 3-7 min/max ranges and normalized trajectories
for all seven calls, including the first call for context.
