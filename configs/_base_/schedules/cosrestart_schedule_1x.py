# learning policy
lr_config = dict(
    policy="CosineRestart",
    warmup="linear",
    periods=[5, 10],
    restart_weights=[0.8, 0.6],
    min_lr_ratio=0.05,
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
