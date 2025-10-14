from typing import List, Any
from dataclasses import field
from pathlib import Path

import sqlalchemy as sa
import omegaconf
import hydra
import hydra_orm.utils
from hydra_orm import orm

from conf import datasets, flow_matching_guidance, inflation_scale, models, diffusion_path, observe
from dafm import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'outputs').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='trajectories')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    dataset = orm.OneToManyField(datasets.Dataset, required=True, default=omegaconf.MISSING)
    model = orm.OneToManyField(models.Model, required=True, default=omegaconf.MISSING)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(datasets.DoubleWell, group=Conf.dataset.key, name=f'_{datasets.DoubleWell.__name__}')
orm.store_config(datasets.Simple, group=Conf.dataset.key, name=f'_{datasets.Simple.__name__}')
orm.store_config(datasets.Lorenz63, group=Conf.dataset.key, name=f'_{datasets.Lorenz63.__name__}')
orm.store_config(datasets.Lorenz96, group=Conf.dataset.key, name=f'_{datasets.Lorenz96.__name__}')
orm.store_config(datasets.NavierStokes, group=Conf.dataset.key, name=f'_{datasets.NavierStokes.__name__}')
orm.store_config(datasets.KuramotoSivashinsky, group=Conf.dataset.key, name=f'_{datasets.KuramotoSivashinsky.__name__}')
orm.store_config(observe.Full, group=f'{Conf.dataset.key}/{datasets.Dataset.observe.key}')
orm.store_config(observe.EveryNthDimension, group=f'{Conf.dataset.key}/{datasets.Dataset.observe.key}')
orm.store_config(observe.Exponentiate, group=f'{Conf.dataset.key}/{datasets.Dataset.observe.key}')
orm.store_config(observe.ATan, group=f'{Conf.dataset.key}/{datasets.Dataset.observe.key}')
orm.store_config(observe.ATanEveryNthDimension, group=f'{Conf.dataset.key}/{datasets.Dataset.observe.key}')
orm.store_config(models.ScoreMatching, group=Conf.model.key, name=f'_{models.ScoreMatching.__name__}')
orm.store_config(models.ScoreMatchingMarginal, group=Conf.model.key, name=f'_{models.ScoreMatchingMarginal.__name__}')
orm.store_config(flow_matching_guidance.No, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}')
orm.store_config(flow_matching_guidance.MonteCarlo, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}', name=f'_{flow_matching_guidance.MonteCarlo.__name__}')
orm.store_config(diffusion_path.ConditionalOptimalTransport, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.PreviousPosteriorToPredictive, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.VarianceExploding, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(flow_matching_guidance.Local, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}', name=f'_{flow_matching_guidance.Local.__name__}')
orm.store_config(flow_matching_guidance.Constant, group=f'{Conf.model.key}/{models.FlowMatching.guidance.key}/{flow_matching_guidance.Local.schedule.key}')
orm.store_config(inflation_scale.NoScaling, group=f'{Conf.model.key}/{models.Model.inflation_scale.key}')
orm.store_config(inflation_scale.ConstantScale, group=f'{Conf.model.key}/{models.Model.inflation_scale.key}')
orm.store_config(models.FlowMatching, group=Conf.model.key, name=f'_{models.FlowMatching.__name__}')
orm.store_config(models.FlowMatchingMarginal, group=Conf.model.key, name=f'_{models.FlowMatchingMarginal.__name__}')
orm.store_config(models.FlowMatchingGaussianTarget, group=Conf.model.key, name=f'_{models.FlowMatchingGaussianTarget.__name__}')
orm.store_config(models.BootstrapParticleFilter, group=Conf.model.key, name=f'_{models.BootstrapParticleFilter.__name__}')
orm.store_config(models.EnsembleKalmanFilterPerturbedObservations, group=Conf.model.key, name=f'_{models.EnsembleKalmanFilterPerturbedObservations.__name__}')
orm.store_config(models.EnsembleKalmanFilterPerturbedObservationsIterative, group=Conf.model.key, name=f'_{models.EnsembleKalmanFilterPerturbedObservationsIterative.__name__}')
orm.store_config(models.EnsembleRandomizedSquareRootFilter, group=Conf.model.key, name=f'_{models.EnsembleRandomizedSquareRootFilter.__name__}')
orm.store_config(models.LocalEnsembleTransformKalmanFilter, group=Conf.model.key, name=f'_{models.LocalEnsembleTransformKalmanFilter.__name__}')
orm.store_config(diffusion_path.ConditionalOptimalTransport, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.PreviousPosteriorToPredictive, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.VarianceExploding, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.Bao2024EnsembleScoreMatching, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')