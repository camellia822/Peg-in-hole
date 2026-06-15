# Peg-in-hole

当前仓库只保留新的 dual-peg peg-in-hole 主线：`pih_rebuild`。

## 目录

```text
pih_rebuild/
	train_sac.py                 # 纯 SAC 训练入口
	config.py                    # 任务参数、奖励参数、扰动参数
	envs/ur5_dual_peg_env.py     # MuJoCo + UR5 IK 双 peg 环境
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
	--tag sac_baseline \
	--obs_mode vision-touch --max_steps 220 \
	--perturb --perturb_intensity 0.65 \
	--timesteps 200000 --seed 7
```

批量对比：

```bash
bash experiments/run_sac_baseline_065_seed7_200k.sh
```

## Acknowledgements, Modifications and Copyright Notice

This repository is developed based on the open-source project SoftBodyInsertion by 0707yiliu.

Original repository:
https://github.com/0707yiliu/SoftBodyInsertion

We sincerely thank the original author(s) for their contribution to the peg-in-hole simulation environment and related implementation.

Compared with the original project, this repository has been reorganized and modified for dual-peg peg-in-hole reinforcement learning experiments. Major modifications include:

- Reorganized the project structure and kept the new main implementation under `pih_rebuild/`.
- Added or modified the UR5 dual-peg peg-in-hole environment.
- Added SAC-based training scripts and experiment scripts.
- Added vision-touch observation mode and related reward / perturbation configurations.
- Removed or deprecated old Gym environment files and older training scripts.
- Added experiment documents and training output management.

This repository is a derivative and modified work based on the original SoftBodyInsertion project. The copyright of the original code belongs to its original author(s). The modifications, restructuring, experiment scripts, and newly added code in this repository are contributed by the current repository maintainer.

If the original project has a license, this repository follows the terms and conditions of the original license for the corresponding derived parts. For files newly created in this repository, the copyright belongs to the current repository maintainer unless otherwise stated.

This repository is used for academic research and learning purposes only. If there is any copyright or license concern, please contact the maintainer, and the relevant content will be corrected or removed promptly.

---

## 致谢、修改内容与版权说明

本仓库是在 0707yiliu 的 SoftBodyInsertion 项目基础上进行学习、复现、整理与二次开发得到的项目。

原始仓库：
https://github.com/0707yiliu/SoftBodyInsertion

感谢原作者对轴孔装配仿真环境和相关代码的开源贡献。

与原项目相比，本仓库主要面向双轴孔装配强化学习实验进行了重构和修改，主要包括：

- 重新整理项目结构，将当前主线代码放在 `pih_rebuild/` 下；
- 新增或修改 UR5 双轴孔装配仿真环境；
- 增加基于 SAC 的训练脚本和批量实验脚本；
- 增加视觉-力觉观测、奖励参数和扰动参数配置；
- 移除或废弃旧版 Gym 环境和旧训练脚本；
- 增加实验说明文档和训练输出管理方式。

本仓库是在原 SoftBodyInsertion 项目基础上进行修改和二次开发得到的衍生项目。原始代码的版权归原作者所有。本仓库中的重构部分、实验脚本、新增代码和修改内容由当前仓库维护者完成。

如果原项目包含开源许可证，本仓库中基于原项目衍生的部分遵循原项目许可证的相关要求；本仓库中新增加的代码，如无特别说明，其版权归当前仓库维护者所有。

本仓库仅用于学术研究和学习交流。如涉及版权或许可证问题，请联系仓库维护者，本人将及时补充说明、修改或移除相关内容。
