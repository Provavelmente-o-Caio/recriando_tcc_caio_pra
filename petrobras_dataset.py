import json
import os
from typing import Any, Dict, List, Literal, Set, Union

import lasio as ls
import polars as pl

LOGS = ["DT", "FACIES", "GR", "NPHI", "DENSIDADE", "MACRO-RESIST"]


def read_single_well(
    file: str,
    with_dept: bool = False,
    features: Union[List[str], Literal["all"]] = LOGS,
) -> pl.DataFrame:
    """Reads a single .las file, filter the columns and returns the DataFrame."""
    directory = "data/petrobras/las_files"

    well = ls.read(f"{directory}/{file}")
    # lasio returns pandas; convert once to keep this module polars-native.
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


def read_all_wells_without_dept_to_list(
    features: Union[List[str], Literal["all"]] = LOGS,
) -> List[pl.DataFrame]:
    """Reads all wells from 'data/' and returns a list of DataFrames with the data.
    By default, gets the logs: ["DT", "FACIES", "GR", "NPHI", "DENSIDADE", "MACRO-RESIST"]
    """
    files = _get_all_raw_files()

    wells: List[pl.DataFrame] = []

    for file_name in files:
        well = read_single_well(file_name, with_dept=False, features=features)

        if features != "all":
            well = well.select(features)

        wells.append(well)

    return wells


def read_all_wells_with_dept_to_list(
    features: Union[List[str], Literal["all"]] = LOGS,
) -> List[pl.DataFrame]:
    """Reads all wells from 'data/' and returns a list of DataFrames with the data, including DEPT.
    By default, gets the logs: ["DT", "FACIES", "GR", "NPHI", "DENSIDADE", "MACRO-RESIST"]
    """
    files = _get_all_raw_files()

    wells_with_dept: List[pl.DataFrame] = []

    for file_name in files:
        well_with_dept = read_single_well(file_name, with_dept=True, features=features)

        if features != "all":
            selected_features = list(features)
            if "DEPT" not in selected_features:
                selected_features.append("DEPT")
            well_with_dept = well_with_dept.select(selected_features)

        wells_with_dept.append(well_with_dept)

    return wells_with_dept


def _get_all_raw_files() -> List[str]:
    """Returns all .las files inside 'las_files' sorted by name."""
    directory = "data/petrobras/las_files"
    all_files = sorted(os.listdir(directory))

    return list(filter(lambda file_name: ".las" in file_name, all_files))


# utilities
def filter_commom_features(
    wells: List[pl.DataFrame], ignore: List[str]
) -> List[pl.DataFrame]:
    """Filters the list of wells to include only the features common to all wells.

    Can receive a list of features to ignore, which will be always present in the wells, even though some wells may not have it
    """
    # Get the set of columns in each well, excluding ignored features
    feature_sets: List[Set[str]] = [set(well.columns) - set(ignore) for well in wells]
    common_features: Set[str] = feature_sets[0].copy()
    for feature_set in feature_sets[1:]:
        common_features.intersection_update(feature_set)

    # Add ignored features back to the set of columns to keep
    final_features = list(common_features.union(ignore))

    # Return the filtered wells
    return [
        well.select([column for column in well.columns if column in final_features])
        for well in wells
    ]


"""
saves the results to a JSON file, with the format:
{
   "experiment_<experiment_number>":{
      "features":[
         "feature1",
         "feature2"
      ],
      "target":"target_feature",
      "train_wells:":[
         "well1",
         "well2"
      ],
      "test_wells":[
         "well3",
         "well4"
      ],
      "model":"model_name",
      "model_params":{
         "param1":"<value>",
         "param2":"<value>"
      },
      "results":{
         "R2":"<value>",
         "RMSE":"<value>",
         "MAE":"<value>"
      },
      "training_time":"value",
      "inference_time":"value",
      "notes":"any additional notes about the experiment"
   }
}

When the file is already present, it will be updated with the new experiment data.
"""


def save_experiment_results(
    experiment_number: int,
    features: List[str],
    target: str,
    train_wells: List[str],
    test_wells: List[str],
    model: str,
    model_params: Dict[str, Any],
    results: Dict[str, Any],
    training_time: float,
    inference_time: float,
    notes: str = "",
) -> None:
    """Saves the experiment results to a JSON file."""

    file_path = "data/petrobras/experiment_results.json"

    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Create the new experiment entry
    experiment_key = f"experiment_{experiment_number}"
    data[experiment_key] = {
        "features": features,
        "target": target,
        "train_wells": train_wells,
        "test_wells": test_wells,
        "model": model,
        "model_params": model_params,
        "results": results,
        "training_time": training_time,
        "inference_time": inference_time,
        "notes": notes,
    }

    # Save the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
