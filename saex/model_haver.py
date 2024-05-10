from .sae import SAEConfig, SAE
from .buffer import ActivationBuffer
from .utils import utils
import equinox as eqx
import jax.sharding as jshard
import jax
import numpy as np
from typing import Optional
from .iterable_dataset import create_iterable_dataset, IterableDatasetConfig


class ModelHaver(object):
    def __init__(self, model_config, sae_config: SAEConfig, dataset_config: IterableDatasetConfig,
                 sae: SAE = None, sae_restore: Optional[str] = None, use_devices=1, mp_devices=1):
        mesh = None

        if model is None:
            print("Loading model...")
            if model_config.model_class.has_mesh:
                model = model_config.model_class(model_config)
                mesh = model.mesh
            else:
                mesh = jshard.Mesh(np.array(jax.devices())[:use_devices].reshape(
                    -1, mp_devices), axis_names=("dp", "mp"))
                model = model_config.model_class(model_config, mesh=mesh)
        self.model = model
        self.mesh = mesh

        if sae is not None:
            self.sae, self.sae_state = utils.unstatify(sae)
        else:
            print("Creating SAE...")
            self.sae, self.sae_state = eqx.nn.make_with_state(SAE)(sae_config, self.mesh)
            if sae_restore:
                print(f"Loading checkpoint ({sae_restore})...")
                self.sae = self.sae.restore(sae_restore)
            sharding = {k: jshard.NamedSharding(self.mesh, v) for k, v in self.sae.get_partition_spec()[0].items()}
            sae_params, _ = eqx.partition(self.sae, lambda x: eqx.is_array(x))
            self.sharding_sae = jax.tree_util.tree_map_with_path(lambda path, x: sharding.get(path[0].name), sae_params)
        
        if create_dataset is None:
            print("Loading dataset...")
            create_dataset = create_iterable_dataset(dataset_config)
        self.create_dataset = create_dataset