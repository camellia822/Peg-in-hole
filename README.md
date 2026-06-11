# SoftBodyInsertion

当前仓库只保留新的 dual-peg peg-in-hole 主线：`pih_rebuild`。

## 目录

```text
pih_rebuild/
	train_sac.py                 # SAC / SAC + M1 / SAC + M1 + M2 训练入口
	config.py                    # 任务参数、奖励参数、扰动参数
	envs/ur5_dual_peg_env.py     # MuJoCo + UR5 IK 双 peg 环境
	spar/                        # M1/M2 算法实现
	robotics/ur5_kdl.py          # UR5 PyKDL 正逆解
	assets/                      # 新主线运行所需 XML、URDF、mesh、texture
experiments/                   # 当前可用实验脚本
docs/                          # 奖励版本和历史说明
output/                        # 训练输出，默认由 .gitignore 忽略
```

旧 `src/`、`gym_envs/`、旧 Gym 环境、旧 PPO/PA-SAC/SCA 脚本已移除。以后实验统一走 `pih_rebuild.train_sac`。

## 验证

```bash
/home/sun/anaconda3/envs/pih_env/bin/python -m pih_rebuild.smoke_test
```

## 训练

SAC baseline:

```bash
/home/sun/anaconda3/envs/pih_env/bin/python -m pih_rebuild.train_sac \
	--algo sac --tag sac_baseline \
	--obs_mode vision-touch --max_steps 220 \
	--perturb --perturb_intensity 0.65 \
	--timesteps 200000 --seed 7
```

SAC + M1:

```bash
/home/sun/anaconda3/envs/pih_env/bin/python -m pih_rebuild.train_sac \
	--algo m1 --tag spar_m1 \
	--obs_mode vision-touch --max_steps 220 \
	--perturb --perturb_intensity 0.65 \
	--timesteps 200000 --seed 7
```

SAC + M1 + M2:

```bash
/home/sun/anaconda3/envs/pih_env/bin/python -m pih_rebuild.train_sac \
	--algo m1m2 --tag spar_m1m2 \
	--obs_mode vision-touch --max_steps 220 \
	--perturb --perturb_intensity 0.65 \
	--timesteps 200000 --seed 7
```

批量对比：

```bash
bash experiments/run_spar_ablation_065_seed7_200k.sh
```
