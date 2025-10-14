from conf import inflation_scale
from omegaconf import DictConfig


class InflationScale:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, sampled_state_centered):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NoScaling(InflationScale):
    pass


class ConstantScale(InflationScale):
    def forward(self, sampled_state_centered):
        return self.cfg.constant


def get_inflation_scale(cfg):
    if isinstance(cfg, inflation_scale.NoScaling):
        return NoScaling(cfg)
    elif isinstance(cfg, inflation_scale.ConstantScale):
        return ConstantScale(cfg)
    else:
        raise ValueError(f'Unknown inflation scale: {cfg}')

def get_inflation_scale(cfg):
    if isinstance(cfg, DictConfig):

        target = cfg.get("_target_", None)
        name = cfg.get("name", None)
        if target == "conf.inflation_scale.ConstantScale" or name == "ConstantScale":
            return ConstantScale(cfg)
        if target == "conf.inflation_scale.NoScaling" or name == "NoScaling":
            return NoScaling(cfg)
        if "constant" in cfg:
            return ConstantScale(cfg)
        raise ValueError(f"Unknown inflation scale DictConfig: {cfg}")

    if isinstance(cfg, inflation_scale.NoScaling):
        return NoScaling(cfg)
    elif isinstance(cfg, inflation_scale.ConstantScale):
        return ConstantScale(cfg)
    else:
        raise ValueError(f'Unknown inflation scale: {cfg}')