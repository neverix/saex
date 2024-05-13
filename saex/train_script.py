from itertools import chain
from typing import List, Union
import gc

import jax_smi

from .trainer_cache import BufferCacher, BufferTrainer, BufferTrainerConfig


def train_main(configs: Union[BufferTrainerConfig, List[BufferTrainerConfig]]):
    if not isinstance(configs, list):
        configs = [configs]

    def cleanup():
        for name in ("cacher", "trainers", "trainer_iterators", "cacher_iterator"):
            try:
                exec(f"del {name}")
            except NameError:
                pass

    try:
        jax_smi.initialise_tracking()
        cacher = BufferCacher(configs[0])
        trainers = []
        for config in configs:
            trainer = BufferTrainer(config, mesh=cacher.mesh, evaluator=cacher)
            trainers.append(trainer)
        trainer_iterators = [trainer.train() for trainer in trainers]
        cacher_iterator = iter(cacher)
        for i in chain(trainer_iterators, [cacher_iterator]):
            next(i)
        while True:
            try:
                batch = cacher_iterator.send(False)
            except StopIteration:
                batch = None
            for iterator in trainer_iterators:
                iterator.send(batch)
            if batch is None:
                break
            del batch
    except:
        cleanup()
        gc.collect()
        raise
    cleanup()
