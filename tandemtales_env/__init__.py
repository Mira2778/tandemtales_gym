from gymnasium.envs.registration import register

register(
    id="tandemtales_env/TandemTalesEnv-v0",
    entry_point="tandemtales_env.envs:TandemTalesEnv",
    max_episode_steps=1000,
)