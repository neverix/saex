from dataclasses import dataclass
from typing import Callable

import datasets


@dataclass
class IterableDatasetConfig:
    dataset_name: str
    split: str = "train"
    text_column: str = "text"
    clean_fn: Callable = lambda x: x


def create_iterable_dataset(config: IterableDatasetConfig):
    dataset = datasets.load_dataset(config.dataset_name, streaming=True)[config.split]
    
    def generator():
        while True:
            for sample in dataset:
                text = config.clean_fn(sample[config.text_column])
                if not text:
                    continue
                yield text
    return generator
