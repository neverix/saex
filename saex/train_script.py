import jax_smi

from .trainer_cache import BufferTrainer, BufferTrainerConfig


def train_main(config: BufferTrainerConfig):
    jax_smi.initialise_tracking()
    trainer = BufferTrainer(config)
    trainer.train()
