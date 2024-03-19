from gymnasium.envs.registration import register


# register new env here
register(
    id='custom_envs/ActiveImageClassifier-v0',
    entry_point='custom_envs.envs:ImageClassifier',
)

register(
    id='custom_envs/ActiveImgFeatureClassifier-v0',
    entry_point='custom_envs.envs:ImgFeatureClassifier',
    max_episode_steps=20_000,
)


register(
    id='custom_envs/ActiveImageClassifier-v1',
    entry_point='custom_envs.envs:ImageClassifier_new',
    # max_episode_steps=20_000,
)
