from typing import Set
from typing import Literal
from typing import Union
from typing import List
import os

import torch
import polars as pl
import lasio as ls

from crossfold_hyperparamet_experiment import CrossFoldHyperparameterExperiment
from petrobras_dataset import filter_commom_features, read_all_wells_with_dept_to_list
from utils.modelTrainer import (
    FinalModelTrainer,
    load_best_configuration,
)
from utils.training_utilities import (
    add_derived_features,
    set_deterministic,
)
from well_similarity_analysis import full_analysis

class Predictor:
    def __init__(self, model_root, device) -> None:
        self.model_root = model_root
        self.device = self._resolve_device(device)
        self.clusters = {}

    def _resolve_device(self, device) -> torch.device:
        if device:
            return torch.device(device)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _read_all_wells_with_dept_to_list(self, features: Union[List[str], Literal["all"]]) -> List[pl.DataFrame]:
        files = self._get_all_raw_files()

        wells_with_dept: List[pl.Dataframe] = []

        for file_name in files:
            well_with_dept = self._read_single_well(file_name, True, features)

            if features != "all":
                selected_features = list(features)
                if "DEPT" not in selected_features:
                    selected_features.append("DEPT")
                well_with_dept = well_with_dept.select(selected_features)

            wells_with_dept.append(well_with_dept)

        return wells_with_dept


    def _get_all_raw_files(self) -> List[str]:
        all_files = sorted(os.listdir(self.model_root))
        print(all_files)

        return list(filter(lambda file_name: ".las" in file_name, all_files))

    def _read_single_well(self, file_name: str, with_dept: bool, features: Union[List[str], Literal["all"]]) -> pl.DataFrame:
        well = ls.read(f"{self.model_root}/{file_name}")
        df = pl.from_pandas(well.df().reset_index())

        if features == "all":
            return df

        selected_features: Set[str] = set(features)
        if with_dept:
            selected_features.add("DEPT")

        drop = [column for column in df.columns if column not in selected_features]
        if drop:
            df = df.drop(drop)

        return df
