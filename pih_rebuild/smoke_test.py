import numpy as np

import pih_rebuild  # noqa: F401
import gym


def run_random_smoke() -> None:
    env = gym.make("UR5DualPegRebuild-v0")
    obs = env.reset(seed=42)
    print("random.obs_shape:", obs.shape)
    print("random.obs_finite:", bool(np.all(np.isfinite(obs))))
    print("random.obs_range:", float(np.min(obs)), float(np.max(obs)))
    for step in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
        print(
            f"random.step={step} reward={reward:.4f} done={done} "
            f"success={info['is_success']:.0f} worst_dist={info['worst_dist']:.5f}"
        )
        if done:
            break


def run_staged_diagnostic() -> None:
    env = gym.make("UR5DualPegRebuild-v0")
    env.reset(seed=0)
    base_env = env.unwrapped
    final_info = None
    for step in range(base_env.config.max_steps):
        left_err, right_err = base_env._point_errors()
        center_err = 0.5 * (left_err + right_err)
        xy_norm = np.linalg.norm(center_err[:2])
        action = np.zeros(3, dtype=np.float64)
        xy_action_scale, z_action_scale, _ = base_env._action_scales()
        action[:2] = np.clip(center_err[:2] / xy_action_scale, -1.0, 1.0)
        if xy_norm < 0.0005:
            action[2] = np.clip(center_err[2] / z_action_scale, -1.0, 1.0)
        obs, reward, done, info = env.step(action)
        final_info = info
        if step % 25 == 0 or done:
            print(
                f"staged.step={step} reward={reward:.4f} done={done} "
                f"success={info['is_success']:.0f} worst_dist={info['worst_dist']:.5f}"
            )
        if done:
            break
    if final_info is None:
        raise RuntimeError("staged diagnostic did not produce any steps")
    print(
        "staged.final "
        f"success={final_info['is_success']:.0f} "
        f"worst_dist={final_info['worst_dist']:.5f} "
        f"reason={final_info.get('done_reason', '')}"
    )


def main() -> None:
    run_random_smoke()
    run_staged_diagnostic()


if __name__ == "__main__":
    main()
