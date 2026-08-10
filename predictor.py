import argparse
import json
import math
import os
import time
from typing import Any, Literal

import lasio as ls
import polars as pl
import torch

from crossfold_hyperparamet_experiment import CrossFoldHyperparameterExperiment
from utils.modelTrainer import (
    FinalModelTrainer,
    load_best_configuration,
)
from utils.training_utilities import (
    add_derived_features,
)
from well_similarity_analysis import (
    WellGroupingRecommendation,
    WellSimilarityAnalyzer,
    WellSimilarityVisualizer,
)

REQUIRED_CURVES = ["VP", "RHO", "GR", "POROSIDADE", "SATURACAO"]


class Predictor:
    def __init__(self, model_root: str, device: None | torch.device) -> None:
        # TODO:
        self.model_root: str = model_root
        self.device: None | torch.device = self._resolve_device(device)
        self.clusters = {}
        # self.wells = self._read_all_wells_with_dept_to_list(features="all")
        # self.wells_dfs = self._filter_common_features(self.wells, ignore=["VS"])

    def _resolve_device(self, device : None | torch.device = None) -> torch.device:
        if device:
            return torch.device(device)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _read_all_wells_with_dept_to_list(
        self, features: list[str] | Literal["all"]
    ) -> list[pl.DataFrame]:
        files = self._get_all_raw_files()

        wells_with_dept: list[pl.DataFrame] = []

        for file_name in files:
            well_with_dept = self._read_single_well(file_name, True, features)

            if features != "all":
                selected_features = list(features)
                if "DEPT" not in selected_features:
                    selected_features.append("DEPT")
                well_with_dept = well_with_dept.select(selected_features)

            wells_with_dept.append(well_with_dept)

        return wells_with_dept

    def _get_all_raw_files(self) -> list[str]:
        all_files = sorted(os.listdir(self.model_root))
        print(all_files)

        return list(filter(lambda file_name: ".las" in file_name, all_files))

    def _read_single_well(
        self,
        file_name: str,
        with_dept: bool,
        features: list[str] | Literal["all"],
    ) -> pl.DataFrame:
        well = ls.read(f"{self.model_root}/{file_name}")
        df = pl.from_pandas(well.df().reset_index())

        if features == "all":
            return df

        selected_features: set[str] = set(features)
        if with_dept:
            selected_features.add("DEPT")

        drop = [column for column in df.columns if column not in selected_features]
        if drop:
            df = df.drop(drop)

        return df

    def _filter_common_features(self, wells: list[pl.DataFrame], ignore: list[str]):
        feature_sets: list[set[str]] = [
            set(well.columns) - set(ignore) for well in wells
        ]
        common_features: set[str] = feature_sets[0].copy()

        for feature_set in feature_sets[1:]:
            common_features.intersection_update(feature_set)

        # Add ignored features back to the set of columns to keep
        final_features = list(common_features.union(ignore))

        # Return the filtered wells
        return [
            well.select([column for column in well.columns if column in final_features])
            for well in wells
        ]

    def prediction_setup(
        self, input_path: str, output_path: str, recommended_clusters_path: str
    ):
        recommended_clusters = load_json_payload(recommended_clusters_path)["clusters"]

        feature_combinations = [
            # More features = better predictions
            ["VP", "RHO", "GR", "CALIPER", "POROSIDADE", "SATURACAO", "ARGILOSIDADE"],
            ["VP", "RHO", "GR", "CALIPER", "POROSIDADE", "SATURACAO"],
            # With derived features (if available)
            ["VP", "RHO", "ACOUSTIC_IMP", "GR", "POROSIDADE", "SATURACAO"],
            # Core physics-based
            ["VP", "RHO", "POROSIDADE", "GR", "ARGILOSIDADE"],
            ["VP", "RHO", "POROSIDADE", "SATURACAO"],
        ]

        # Load payload and validate curves/wells
        payload = load_json_payload(input_path)

        curve_mapping = validate_curve_mapping(payload)
        wells = validate_wells(payload)
        config = validate_config(payload)

        base_config = {
            "sequence_length": config.get("SequenceLength"),
            "mask_value": config.get("MaskValue"),
            "num_epochs": config.get("NumEpochs"),
            "patience": config.get("Patience"),
            "target_feature": config.get("TargetFeature"),
            "clusters": recommended_clusters,
        }

        wells_dfs = wells_to_dataframes(wells, curve_mapping)

        wells_dfs = [add_derived_features(df) for df in wells_dfs]

        wells_with_vs = [df for df in wells_dfs if "VS" in df.columns]
        wells_without_vs = [df for df in wells_dfs if "VS" not in df.columns]

        return (
            base_config,
            feature_combinations,
            wells_with_vs,
            wells_without_vs,
        )

    def analyze(self, input_path: str, output_path: str):
        """
        Main execution function.

        Workflow:
        1. Load well data
        2. Compute all similarity metrics
        3. Visualize similarities
        4. Generate recommendations
        """
        payload = load_json_payload(input_path)

        curve_mapping = validate_curve_mapping(payload)
        wells = validate_wells(payload)

        wells_dfs = wells_to_dataframes(wells, curve_mapping)
        wells_dfs = [add_derived_features(df) for df in wells_dfs]

        wells_with_vs = [df for df in wells_dfs if "VS" in df.columns]

        # Define features to analyze (use what you have)
        features_to_analyze = ["VP", "RHO", "POROSIDADE", "SATURACAO"]

        for feature in features_to_analyze:
            for index, df in enumerate(wells_with_vs):
                if feature not in df.columns:
                    raise ValueError(
                        f"Well index {index} is missing feature '{feature}'."
                    )

        analyzer = WellSimilarityAnalyzer(wells_with_vs, features_to_analyze)
        _ = analyzer.compute_all_distances()

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plots_dir = (
            os.path.join(output_dir, "well_similarity_analysis")
            if output_dir
            else "well_similarity_analysis"
        )

        visualizer = WellSimilarityVisualizer(analyzer, plots_dir)
        visualizer.create_all_plots()

        recommender = WellGroupingRecommendation(analyzer)
        recommender.generate_summary_report()
        recommended_clusters = recommender.identify_well_clusters(cluster_height=1)

        output_dir = os.path.join(output_dir, "well_similarity_analysis")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        result = {
            "status": "success",
            "clusters": {
                str(cluster): list(map(int, indices))
                for cluster, indices in recommended_clusters.items()
            },
            "visualizations": {
                "Dendrogram combined": os.path.join(
                    output_dir, "dendrogram_combined.png"
                ),
                "Heatmap combined": os.path.join(output_dir, "heatmap_combined.png"),
                "Heatmap correlation": os.path.join(
                    output_dir, "heatmap_correlation.png"
                ),
                "Heatmap DTW": os.path.join(output_dir, "heatmap_dtw.png"),
                "Heatmap feature space": os.path.join(
                    output_dir, "heatmap_feature_space.png"
                ),
                "Heatmap statistical": os.path.join(
                    output_dir, "heatmap_statistical.png"
                ),
                "MDS combined": os.path.join(output_dir, "mds_combined.png"),
                "MDS comparison": os.path.join(output_dir, "mds_comparison.png"),
                "MDS correlation": os.path.join(output_dir, "mds_correlation.png"),
                "MDS DTW": os.path.join(output_dir, "mds_dtw.png"),
                "MDS feature space": os.path.join(output_dir, "mds_feature_space.png"),
                "MDS statistical": os.path.join(output_dir, "mds_statistical.png"),
            },
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

    def predict(
        self,
        input_path: str,
        output_path: str,
        recommended_clusters_path: str,
        experiment_dir: str,
    ):
        (
            base_config,
            _,
            wells_with_vs,
            wells_without_vs,
        ) = self.prediction_setup(input_path, output_path, recommended_clusters_path)

        # ## Step 2: Run Cross-Fold Validation
        best_config = load_best_configuration(experiment_dir)

        # ## STEP 3: Load or Train Final Model
        final_output_dir = os.path.join(experiment_dir, "final_model")

        trainer = FinalModelTrainer(
            best_config, base_config, output_dir=final_output_dir
        )
        train_start = time.perf_counter()
        trained_clusters = trainer.load_final_models(
            wells_with_vs, target_feature=base_config["target_feature"]
        )
        if not trained_clusters:
            print("\nNo saved final models found; training from scratch...")
            trained_clusters = trainer.train_final_model(
                wells_with_vs, target_feature=base_config["target_feature"]
            )
        train_elapsed = time.perf_counter() - train_start
        print(
            f"\n[PERF] load/train final model took {train_elapsed:.1f}s "
            f"({train_elapsed / 60:.1f} min)"
        )

        # ## STEP 4: Make Predictions
        predict_start = time.perf_counter()
        all_results = trainer.predict_on_wells(
            trained_clusters,
            wells_without_vs,
            wells_with_vs,
            target_feature=base_config["target_feature"],
        )
        predict_elapsed = time.perf_counter() - predict_start
        print(
            f"\n[PERF] predict_on_wells took {predict_elapsed:.1f}s "
            f"({predict_elapsed / 60:.1f} min)"
        )

        # ## STEP 5: Generate Plots and Reports
        plot_start = time.perf_counter()
        trainer.plot_predictions(all_results)
        trainer.generate_summary_report(all_results, best_config)
        plot_elapsed = time.perf_counter() - plot_start
        print(
            f"\n[PERF] plots + summary report took {plot_elapsed:.1f}s "
            f"({plot_elapsed / 60:.1f} min)"
        )

    def train(self, input_path: str, output_path: str, recommended_clusters_path: str):
        # JANK: THIS IS A TEMPORARY FIX
        working_dir = os.path.dirname(output_path)
        working_dir = working_dir.removesuffix("cluster_analysis_output.json")
        experiments_dir = os.path.join(working_dir, "experiments")

        if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)
        (
            base_config,
            feature_combinations,
            wells_with_vs,
            wells_without_vs,
        ) = self.prediction_setup(input_path, output_path, recommended_clusters_path)

        # print("\nRunning cross-fold validation...")
        experiment_dir = self._run_new_experiment(
            base_config, feature_combinations, wells_with_vs, experiments_dir
        )

        best_config = load_best_configuration(experiment_dir)

        final_output_dir = os.path.join(experiment_dir, "final_model")
        trainer = FinalModelTrainer(
            best_config, base_config, output_dir=final_output_dir
        )
        trained_clusters = trainer.train_final_model(
            wells_with_vs, target_feature=base_config["target_feature"]
        )

        all_results = trainer.predict_on_wells(
            trained_clusters,
            wells_without_vs,
            wells_with_vs,
            target_feature=base_config["target_feature"],
        )

        trainer.plot_predictions(all_results)
        trainer.generate_summary_report(all_results, best_config)

    def _run_new_experiment(
        self, base_config, feature_combinations, wells_with_vs, results_dir
    ):
        experiment = CrossFoldHyperparameterExperiment(base_config, results_dir)
        _ = experiment.run_cross_fold_experiments(feature_combinations, wells_with_vs)
        experiment_dir = experiment.results_dir

        return experiment_dir

    def _run_existing_experiment(self, existing_experiments: list[str]):
        if len(existing_experiments) == 1:
            experiment_dir = existing_experiments[0]
        else:
            exp_idx = (
                int(
                    input(
                        f"Which experiment? (1-{len(existing_experiments)}): "
                    ).strip()
                )
                - 1
            )
            experiment_dir = existing_experiments[exp_idx]
        return experiment_dir


def load_json_payload(input_path: str) -> dict[str, Any]:
    if not input_path:
        raise ValueError("Input path is required")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a dictionary")

    return payload


def validate_curve_mapping(payload: dict[str, Any]) -> dict[str, str]:
    mapping = payload.get("curveMapping")

    if not isinstance(mapping, dict):
        raise ValueError(f"curveMapping must be a dictionary: {mapping}")

    for canonical_name in REQUIRED_CURVES:
        selected_name = mapping.get(canonical_name)

        if selected_name.strip() is None or not isinstance(selected_name, str):
            raise ValueError(f"Missing curve: {canonical_name}")

    return {key: str(value).strip() for key, value in mapping.items()}


def validate_wells(payload: dict[str, Any]):
    wells = payload.get("wells")

    if not isinstance(wells, list) or len(wells) == 0:
        raise ValueError(f"wells must be a list: {wells}")

    for index, well in enumerate(wells):
        if not isinstance(well, dict):
            raise ValueError(f"well at index {index} must be a dictionary: {well}")

        name = well.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"well at index {index} must have a non-empty name: {name}"
            )

        logs = well.get("logs")
        if not isinstance(logs, list) or len(logs) == 0:
            raise ValueError(f"well at index {index} must have logs: {logs}")

    return wells


def validate_config(payload: dict[str, Any]):
    config = payload.get("pythonConfiguration")

    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary: {config}")

    return config


def validate_log_structure(well: dict[str, Any]):
    for index, log in enumerate(well["logs"]):
        if not isinstance(log, dict):
            raise ValueError(f"log at index {index} must be a dictionary: {log}")

        log_name = log.get("name")
        if not isinstance(log_name, str) or not log_name.strip():
            raise ValueError(f"log at index {index} must have a non-empty name")

        samples = log.get("samples")
        if not isinstance(samples, list) or len(samples) == 0:
            raise ValueError(f"log at index {index} must have samples")


def wells_to_dataframes(
    wells: list[dict[str, Any]],
    curve_mapping: dict[str, str],
) -> list[pl.DataFrame]:
    dataframes: list[pl.DataFrame] = []

    inverse_mapping = {
        petrel_name: canonical_name
        for canonical_name, petrel_name in curve_mapping.items()
        if canonical_name != "OUTPUT_CURVE"
    }

    for well in wells:
        validate_log_structure(well)

        rows_by_md: dict[float, dict[str, float]] = {}

        for log in well["logs"]:
            petrel_log_name = log["name"]

            if petrel_log_name not in inverse_mapping:
                continue

            canonical_name = inverse_mapping[petrel_log_name]

            for sample in log["samples"]:
                if not isinstance(sample, dict):
                    continue

                md = sample.get("md")
                value = sample.get("value")

                if md is None or value is None:
                    continue

                try:
                    md_float = float(md)
                    value_float = float(value)
                except TypeError, ValueError:
                    continue

                if math.isnan(md_float) or math.isnan(value_float):
                    continue

                if md_float not in rows_by_md:
                    rows_by_md[md_float] = {"DEPT": md_float}

                rows_by_md[md_float][canonical_name] = value_float

        rows = list(rows_by_md.values())
        rows.sort(key=lambda row: row["DEPT"])

        if not rows:
            raise ValueError(
                f"Well '{well['name']}' produced no valid rows. Original rows: {rows_by_md}"
            )

        df = pl.DataFrame(rows)

        missing_features = [
            feature for feature in REQUIRED_CURVES if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Well '{well['name']}' is missing mapped curves: {missing_features}"
            )

        dataframes.append(df)

    return dataframes


def main():
    # var arguments =
    #     "\"" + runnerPath + "\" " +
    #     mode + " " +
    #     "--input \"" + inputPath + "\" " +
    #     "--output \"" + outputPath + "\"";

    # AppendStatus("Running: " + pythonExe + " " + arguments);
    # AppendStatus("Working directory: " + workingDirectory);
    #
    # [14:34:10] Running: C:\Users\caiof\AppData\Roaming\recriando_tcc_caio_pra\.venv\Scripts\python.exe "C:\Users\caiof\AppData\Roaming\recriando_tcc_caio_pra\predictor.py" analyze --input "C:\Users\caiof\AppData\Local\Temp\vs_predictior_petrel\cluster_analysis_input.json" --output "C:\Users\caiof\AppData\Local\Temp\vs_predictior_petrel\cluster_analysis_output.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["analyze", "train", "predict"])
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--cluster", type=str, help="Cluster for your selected well log"
    )
    parser.add_argument("--experiment_dir", type=str, help="Experiment directory")
    args = parser.parse_args()

    predictor = Predictor(
        model_root="C:\\Users\\caiof\\AppData\\Roaming\\recriando_tcc_caio_pra\\data\\petrobras",
        device=None,
    )

    match args.mode:
        case "analyze":
            predictor.analyze(args.input, args.output)
        case "train":
            predictor.train(args.input, args.output, args.cluster)
        case "predict":
            predictor.predict(
                args.input, args.output, args.cluster, args.experiment_dir
            )


if __name__ == "__main__":
    main()
