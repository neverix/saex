import gc
from itertools import chain
from typing import List, Union

import jax
import jax_smi

import wandb

from .trainer_cache import BufferCacher, BufferTrainer, BufferTrainerConfig


def train_main(configs: Union[BufferTrainerConfig, List[BufferTrainerConfig]]):
    if not isinstance(configs, list):
        configs = [configs]
    config = configs[0]

    def cleanup():
        for name in ("cacher", "trainers", "trainer_iterators", "cacher_iterator",
                     "iterator", "batch"):
            try:
                exec(f"del {name}")
            except NameError:
                pass
        if config.is_distributed:
            jax.distributed.shutdown()
        gc.collect()

    try:
        if config.is_distributed:
            jax.distributed.initialize()
        jax_smi.initialise_tracking()
        if config.use_wandb:
            wandb.init(entity=config.use_wandb[0], project=config.use_wandb[1])
        cacher = BufferCacher(config)
        trainers = []
        for config in configs:
            trainer = BufferTrainer(config, mesh=cacher.mesh, evaluator=cacher)
            trainers.append(trainer)
        trainer_iterators = [trainer.train(wandb_suffix=str(i)) for i, trainer in enumerate(trainers)]
        cacher_iterator = iter(cacher)
        for i in chain(trainer_iterators, [cacher_iterator]):
            next(i)
        while True:
            import time
            time_start = time.time()
            try:
                batch = cacher_iterator.send(False)
            except StopIteration:
                batch = None
            time_end = time.time()
            for iterator in trainer_iterators:
                time_start = time.time()
                try:
                    iterator.send(batch)
                except StopIteration:
                    continue
                time_end = time.time()
            if batch is None:
                break
            del batch
    except:
        cleanup()
        raise
    cleanup()
