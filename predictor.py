from typing import Set
from typing import Literal
from typing import Union
from typing import List
import os

import torch
import polars as pl
import lasio as ls

from crossfold_hyperparamet_experiment import CrossFoldHyperparameterExperiment
from utils.modelTrainer import (
    FinalModelTrainer,
    load_best_configuration,
)
from utils.training_utilities import (
    add_derived_features,
)
from well_similarity_analysis import (
    WellSimilarityAnalyzer,
    WellSimilarityVisualizer,
    WellGroupingRecommendation,
)


class Predictor:
    def __init__(self, model_root, device) -> None:
        self.model_root = model_root
        self.device = self._resolve_device(device)
        self.clusters = {}
        self.wells = self._read_all_wells_with_dept_to_list(features="all")
        self.wells_dfs = self._filter_common_features(self.wells, ignore=["VS"])

    def _resolve_device(self, device) -> torch.device:
        if device:
            return torch.device(device)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _read_all_wells_with_dept_to_list(
        self, features: Union[List[str], Literal["all"]]
    ) -> List[pl.DataFrame]:
        files = self._get_all_raw_files()

        wells_with_dept: List[pl.DataFrame] = []

        for file_name in files:
            well_with_dept = self._read_single_well(file_name, True, features)

            if features != "all":
                selected_features = list(features)
                if "DEPT" not in selected_features:
                    selected_features.append("DEPT")
                well_with_dept = well_with_dept.select(selected_features)

            wells_with_dept.append(well_with_dept)

        return wells_with_dept

    def full_analysis(self, cluster_height: int = 2):
        """
        Main execution function.

        Workflow:
        1. Load well data
        2. Compute all similarity metrics
        3. Visualize similarities
        4. Generate recommendations
        """
        print("\nSTEP 1: Loading well data")
        # Filter wells with VS (for analysis)
        wells_with_vs = [df for df in self.wells_dfs if "VS" in df.columns]

        print(f"Loaded {len(wells_with_vs)} wells with VS data")

        # Define features to analyze (use what you have)
        features_to_analyze = ["VP", "RHO", "POROSIDADE", "SATURACAO"]

        print("\nSTEP 2: Computing well similarities...")

        analyzer = WellSimilarityAnalyzer(wells_with_vs, features_to_analyze)
        _ = analyzer.compute_all_distances()

        generate_visualization = (
            input("Do you want to generate a visualization? (y/n): ") == "y"
        )
        if generate_visualization:
            print("\nSTEP 3: Creating visualizations...")

            visualizer = WellSimilarityVisualizer(analyzer)
            visualizer.create_all_plots()
        else:
            print("\nSTEP 3: Skipping visualization creation.")

        print("\nSTEP 4: Generating recommendations...")

        recommender = WellGroupingRecommendation(analyzer)
        recommender.generate_summary_report()
        recommended_clusters = recommender.identify_well_clusters(cluster_height=cluster_height)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        if generate_visualization:
            print("\nAll visualizations saved to: well_similarity_analysis/")
            print("\nKey outputs:")
            print("  - MDS plots: Show 2D well relationships")
            print("  - Dendrograms: Show hierarchical grouping")
            print("  - Heatmaps: Show pairwise distances")
            print("  - Console output: Recommendations and insights")
            print("\n" + "=" * 80)

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print(recommender.identify_well_clusters())
        print("=" * 80)

        return recommended_clusters

    def _get_all_raw_files(self) -> List[str]:
        all_files = sorted(os.listdir(self.model_root))
        print(all_files)

        return list(filter(lambda file_name: ".las" in file_name, all_files))

    def _read_single_well(
        self,
        file_name: str,
        with_dept: bool,
        features: Union[List[str], Literal["all"]],
    ) -> pl.DataFrame:
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

    def _filter_common_features(self, wells: List[pl.DataFrame], ignore: List[str]):
        feature_sets: List[Set[str]] = [
            set(well.columns) - set(ignore) for well in wells
        ]
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

    def predict(self):
        recommendded_clusters = self.full_analysis()

        base_config = {
            "sequence_length": 15,
            "mask_value": -1.0,
            "num_epochs": 500,
            "patience": 150,
            "target_feature": "VS",
            "clusters": recommendded_clusters,
        }

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

        self.wells_dfs = [add_derived_features(df) for df in self.wells_dfs]

        wells_with_vs = [df for df in self.wells_dfs if "VS" in df.columns]
        wells_without_vs = [df for df in self.wells_dfs if "VS" not in df.columns]

        print(f"Total wells loaded: {len(self.wells_dfs)}")
        print(f"Wells with VS: {len(wells_with_vs)}")
        print(f"Wells without VS: {len(wells_without_vs)}")

        print("STEP 2: CROSS-FOLD VALIDATION")

        existing_experiments = [
            d for d in os.listdir("experiments") if d.startswith("experiment_")
        ]

        if existing_experiments:
            print(f"\nFound {len(existing_experiments)} existing experiment(s):")
            for i, exp_dir in enumerate(existing_experiments):
                print(f" {i + 1}. {exp_dir}")

            use_existing_experiments = (
                input("\nDo you want to use existing experiment(s) (y/n): ")
                .strip()
                .lower()
                == "y"
            )

            if use_existing_experiments:
                experiment_dir = self._run_existing_experiment(existing_experiments)
                print(f"\nUsing existing experiment: {experiment_dir}")
            else:
                print("\nRunning new cross-fold validation")
                experiment_dir = self._run_new_experiment(
                    base_config, feature_combinations, wells_with_vs
                )

        else:
            print("\nNo existing experiments found. Running cross-fold validation...")
            experiment_dir = self._run_new_experiment(
                base_config, feature_combinations, wells_with_vs
            )

        best_config = load_best_configuration(experiment_dir)

        print("STEP 3: TRAINING FINAL MODEL")

        final_output_dir = os.path.join(experiment_dir, "final_model")
        trainer = FinalModelTrainer(
            best_config, base_config, output_dir=final_output_dir
        )
        trained_clusters = trainer.train_final_model(
            wells_with_vs, target_feature=base_config["target_feature"]
        )

        print("SETP 4: MAKING PREDICTIONS")

        all_results = trainer.predict_on_wells(
            trained_clusters,
            wells_without_vs,
            wells_with_vs,
            target_feature=base_config["target_feature"],
        )

        print("STEP 5: GENERATING REPORTS AND PLOTS")

        trainer.plot_predictions(all_results)
        trainer.generate_summary_report(all_results, best_config)

        # # Done
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nAll results saved to: {trainer.output_dir}/")
        print("  - Final model: final_model.pth")
        print("  - Scaler: final_scaler.pkl")
        print("  - Predictions: all_predictions.json")
        print("  - Summary: SUMMARY_REPORT.txt")
        print("  - Plots: plots/")
        print("\n" + "=" * 80)

    def _run_new_experiment(self, base_config, feature_combinations, wells_with_vs):
        experiment = CrossFoldHyperparameterExperiment(base_config)
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

def main():
    predictor = Predictor(model_root="data/petrobras/las_files", device=None)
    predictor.predict()

if __name__ == "__main__":
    main()
