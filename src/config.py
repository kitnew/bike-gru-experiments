from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional, List, Tuple

@dataclass
class DataConfig:
    dataset_name: str
    dataset_url: str
    seed: int

@dataclass
class DirsConfig:
    raw_dir: Path
    processed_dir: Path
    checkpoint_dir: Path
    logs_dir: Path
    plots_dir: Path

@dataclass
class normalization:
    method: str
    feature_range: List[float]

@dataclass
class validation:
    check_missing: bool
    check_dimensions: bool

@dataclass
class output:
    format: str

@dataclass
class PreprocessConfig:
    seq_len: int
    horizon: int
    train_split: List[float]
    target_idx: List[int]
    datetime_format: str
    datetime_column: str
    normalization: normalization
    validation: validation
    output: output

@dataclass
class DataloaderConfig:
    batch_size: int
    num_workers: int
    shuffle: bool

@dataclass
class Config:
    data: DataConfig
    dirs : DirsConfig
    preprocess: PreprocessConfig
    dataloader: DataloaderConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(
            dataset_name=config_dict['data']['dataset_name'],
            dataset_url=config_dict['data']['dataset_url'],
            seed=int(config_dict['data']['seed'])
        )
        
        dirs_config = DirsConfig(
            raw_dir=Path(config_dict['dirs']['raw_dir']),
            processed_dir=Path(config_dict['dirs']['processed_dir']),
            checkpoint_dir=Path(config_dict['dirs']['checkpoint_dir']),
            logs_dir=Path(config_dict['dirs']['logs_dir']),
            plots_dir=Path(config_dict['dirs']['plots_dir'])
        )

        preprocess_config = PreprocessConfig(
            seq_len=config_dict['preprocess']['seq_len'],
            horizon=config_dict['preprocess']['horizon'],
            train_split=config_dict['preprocess']['train_split'],
            target_idx=config_dict['preprocess']['target_idx'],
            datetime_format=config_dict['preprocess']['datetime_format'],
            datetime_column=config_dict['preprocess']['datetime_column'],
            normalization=normalization(
                method=config_dict['preprocess']['normalization']['method'],
                feature_range=config_dict['preprocess']['normalization']['feature_range']
            ),
            validation=validation(
                check_missing=config_dict['preprocess']['validation']['check_missing'],
                check_dimensions=config_dict['preprocess']['validation']['check_dimensions']
            ),
            output=output(
                format=config_dict['preprocess']['output']['format']
            )
        )

        dataloader_config = DataloaderConfig(
            batch_size=config_dict['dataloader']['batch_size'],
            num_workers=config_dict['dataloader']['num_workers'],
            shuffle=config_dict['dataloader']['shuffle']
        )
        
        return cls(
            data=data_config,
            dirs=dirs_config,
            preprocess=preprocess_config,
            dataloader=dataloader_config
        )

config = Config.from_yaml("/home/kitne/University/2lvl/SU/bike-gru-experiments/config/default.yaml")