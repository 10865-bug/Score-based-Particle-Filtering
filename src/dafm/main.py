import logging
import pprint
import sys
import uuid
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import lightning.pytorch as pl
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

from conf import conf
from conf.conf import Conf
from dafm import callbacks, datasets, loggers, models, utils

log = logging.getLogger(__file__)


class DataAssimilation(pl.LightningModule):
    def __init__(self, cfg, dataset, model):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_iterable = None
        self.model = model

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        if stage == 'fit':
            self.dataset_iterable = iter(self.dataset)

    def train_dataloader(self):
        epoch_count, time_step, time, next_predicted_state, next_observation, ignore_observation = next(self.dataset_iterable)
        self.optimizer = self.model.get_optimizer(time_step, ignore_observation)
        return CombinedLoader({
            epoch: iter(CombinedLoader(dict(
                time_step=DataLoader([time_step]),
                time=DataLoader([time]),
                next_predicted_state=DataLoader(
                    next_predicted_state,
                    batch_size=self.cfg.model.batch_size,
                    shuffle=self.cfg.model.shuffle_training_samples
                ),
                next_observation=DataLoader([next_observation]),
                ignore_observation=DataLoader([ignore_observation]),
            ), mode='max_size_cycle'))
            for epoch in range(epoch_count)
        }, mode='sequential')

    def training_step(self, batch, _):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        self.optimizer.zero_grad()
        next_observation = batch['next_observation'] if not batch['ignore_observation'] else None
        losses = self.model.loss(batch['next_predicted_state'], next_observation, self.dataset.dataset.observe)
        self.manual_backward(losses['loss'])
        self.optimizer.step()
        return losses


@hydra.main(
    version_base=None,
    config_name="Conf", 
)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg_for_db = OmegaConf.to_container(cfg, resolve=True)
        conf.orm.instantiate_and_insert_config(db, cfg_for_db)
        db.commit()

    log.info('Command: python %s', ' '.join(sys.argv))
    log.info(pprint.pformat(OmegaConf.to_container(cfg, resolve=False)))

    alt_id = str(uuid.uuid4())[:8]
    run_dir = Path(cfg.out_dir) / cfg.run_subdir / alt_id
    log.info('Output directory: %s', run_dir)
    
    dataset_cfg_obj = instantiate(cfg.dataset)

    rng = np.random.default_rng(utils.RNG_RANDBITS[cfg.rng_seed])
    dynamics = datasets.get_dynamics_dataset(dataset_cfg_obj, rng, cfg.device, delete_true_state=True)

    pl.seed_everything(cfg.rng_seed)
    with pl.utilities.seed.isolate_rng():
        model_cfg_obj = instantiate(cfg.model)
        model = models.get_model(
            model_cfg_obj,
            cfg.dataset.state_dimension,
            cfg.dataset.observation_noise_std,
            dynamics
        )

    dataset_logger = loggers.CSVLogger(run_dir, name=None, name_metrics_file='dataset_metrics.csv')
    dataset = datasets.PredictedStatesAndObservation(
        dynamics, model,
        logger=dataset_logger,
        save_data=False,
        data_to_save_callback=lambda time_step, data_to_save: datasets.save_trajectories(
            cfg.dataset, data_to_save,
            run_dir / (
                f'{cfg.prediction_filename}.{time_step}.parquet'
                if cfg.dataset.save_data_every_n_time_steps is not None
                else f'{cfg.prediction_filename}.parquet'
            )
        )
    )
    data_assimilation = DataAssimilation(cfg, dataset, model)

    logger = loggers.CSVLogger(run_dir, name=None)
    cbs = [callbacks.LogStats()]
    enable_progress_bar = (cfg.model.epoch_count > 0)
    if enable_progress_bar:
        cbs.append(callbacks.TimeStepProgressBar(cfg))

    trainer = pl.Trainer(
        enable_progress_bar=enable_progress_bar,
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_epochs=-1,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
    )

    try:
        trainer.fit(data_assimilation)
    except StopIteration as e:
        if cfg.model.epoch_count == 0 and cfg.model.epoch_count_sampling == 0:
            pass
        else:
            raise e


if __name__ == '__main__':
    main()
