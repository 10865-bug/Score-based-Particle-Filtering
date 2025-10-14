from conf import observe as cfg_observe
from omegaconf import DictConfig
import inspect
from hydra.utils import instantiate

class Observe:
    def __init__(self, cfg, cfg_dataset):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset

    def forward(self, state):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Full(Observe):
    def forward(self, state):
        return state
    
    def observe(self, state):
        return self.forward(state)

    def __call__(self, state):
        return self.forward(state)


class EveryNthDimension(Observe):
    def __init__(self, cfg, cfg_dataset):
        super().__init__(cfg, cfg_dataset)
        if not cfg.start_at_zero and cfg_dataset.state_dimension < cfg.n:
            raise ValueError(
                f'Cannot observe dimension {cfg.n} of a dataset with only {cfg_dataset.state_dimension} dimensions.'
                f' Please either set dataset.observe.start_at_zero=true to include dimension zero, or set dataset.observe.n to be between 1 and {cfg_dataset.state_dimension} (inclusive).'
            )

    def forward(self, state):
        if self.cfg.start_at_zero:
            return state[..., ::self.cfg.n]
        else:
            return state[..., 1::self.cfg.n]


class Exponentiate(Observe):
    def forward(self, state):
        return state**self.cfg.exponent


class ATan(Observe):
    def forward(self, state):
        return state.atan()


class ATanEveryNthDimension(EveryNthDimension):
    def forward(self, state):
        return super().forward(state).atan()


RUNTIME_MAP = {
    "conf.observe.Full": Full,
    "conf.observe.EveryNthDimension": EveryNthDimension,
    "conf.observe.ATan": ATan,
    "conf.observe.ATanEveryNthDimension": ATanEveryNthDimension,
    "conf.observe.Exponentiate": Exponentiate,
}

def get_observe(cfg):
    obs = getattr(cfg, "observe", None)

    if isinstance(obs, (DictConfig, dict)):
        target = obs.get("_target_")
        runtime_cls = RUNTIME_MAP.get(target)
        if runtime_cls is None:
            raise ValueError(f"Unknown observe target: {target}")
        inst = runtime_cls(cfg.observe, cfg) 
        return inst

    if isinstance(obs, str):
        name = obs
        target = f"conf.observe.{name}"
        runtime_cls = RUNTIME_MAP.get(target)
        if runtime_cls is None:
            raise ValueError(f"Unknown observe string: {obs}")
        inst = runtime_cls(cfg.observe, cfg)
        return inst

    if inspect.isclass(obs):
        target = f"{obs.__module__}.{obs.__name__}"
        runtime_cls = RUNTIME_MAP.get(target)
        if runtime_cls is None:
            if obs in (Full, EveryNthDimension, ATan, ATanEveryNthDimension, Exponentiate):
                inst = obs(cfg.observe, cfg)
                return inst
            raise ValueError(f"Unknown observe class: {target}")
        inst = runtime_cls(cfg.observe, cfg)
        return inst

    if isinstance(obs, (Full, EveryNthDimension, ATan, ATanEveryNthDimension, Exponentiate)):
        return obs

    if isinstance(obs, (cfg_observe.Full,
                        cfg_observe.EveryNthDimension,
                        cfg_observe.ATan,
                        cfg_observe.ATanEveryNthDimension,
                        cfg_observe.Exponentiate)):
        target = f"{obs.__class__.__module__}.{obs.__class__.__name__}"
        runtime_cls = RUNTIME_MAP.get(target)
        if runtime_cls is None:
            raise ValueError(f"Unknown observe instance of config class: {target}")
        inst = runtime_cls(cfg.observe, cfg)
        return inst
    if callable(obs):
        return obs

    raise TypeError(f"Unknown observe: {obs} ({type(obs)})")