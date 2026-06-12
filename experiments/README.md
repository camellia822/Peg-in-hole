# Experiments

Current entry point:

```bash
/home/sun/anaconda3/envs/pih_env/bin/python -m pih_rebuild.train_sac
```

Main comparison:

```bash
bash experiments/run_sac_baseline_065_seed7_200k.sh
```

Single-factor perturbation sweeps:

- `run_perturb_sweep.sh`
- `run_ablation_vbias_static_3x3.sh`
- `run_ablation_vnoise_static_3x3.sh`
- `run_ablation_vocc_static_3x3.sh`
- `run_ablation_fnoise_static_3x3.sh`
- `run_ablation_fdrift_static_3x3.sh`
