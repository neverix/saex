from dataclasses import dataclass

import datasets


@dataclass
class IterableDatasetConfig:
    dataset_name: str
    split: str = "train"
    text_column: str = "text"


def create_iterable_dataset(config: IterableDatasetConfig):
    dataset = datasets.load_dataset(config.dataset_name, streaming=True)[config.split]
    
    def generator():
        for sample in dataset:
            yield sample[config.text_column]
    return generator
