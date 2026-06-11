from gym.envs.registration import register


register(
    id="UR5DualPegRebuild-v0",
    entry_point="pih_rebuild.envs:UR5DualPegEnv",
    max_episode_steps=220,
)
