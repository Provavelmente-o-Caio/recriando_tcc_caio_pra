import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from crossfold_hyperparamet_experiment import CrossFoldHyperparameterExperiment
from execution_contract import (
    emit_progress,
    write_execution_contract,
    write_progress_contract,
)
from petrobras_dataset import filter_commom_features, read_all_wells_with_dept_to_list
from predictor import Predictor
from selfAttention import Rebuilt_SAIDNN
from utils.modelTrainer import FinalModelTrainer, load_best_configuration
from utils.training_utilities import (
    WarmupScheduler,
    add_derived_features,
    evaluate_model,
    set_deterministic,
    train_model_with_validation_split,
)
from utils.WellLogDataset import WellLogDataset, create_augmentation


def normalize_clusters_for_labeled_wells(
    clusters, labeled_well_indices, labeled_well_count
):
    """Convert payload well indices to the compact labeled-well index space."""
    payload_to_labeled = {
        int(payload_index): labeled_index
        for labeled_index, payload_index in enumerate(labeled_well_indices)
    }
    normalized = {}

    for cluster_name, cluster_indices in clusters.items():
        indices = [int(index) for index in cluster_indices]
        if all(0 <= index < labeled_well_count for index in indices):
            normalized[cluster_name] = sorted(set(indices))
            continue

        mapped = [
            payload_to_labeled[index]
            for index in indices
            if index in payload_to_labeled
        ]
        if not mapped:
            raise ValueError(
                f"Cluster '{cluster_name}' has no wells with the target feature "
                "after converting payload indices."
            )

        print(
            f"Normalized cluster '{cluster_name}' from payload well indices "
            "to labeled-well indices."
        )
        normalized[cluster_name] = sorted(set(mapped))

    return normalized


class OptunaCrossFoldExperiment(CrossFoldHyperparameterExperiment):
    """Same leave-one-well-out folds as the parent, but the fixed
    hyperparameter grid is replaced by an Optuna study per cluster."""

    def __init__(self, *args, progress_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = progress_callback

    def run_single_experiment(
        self,
        features_to_use,
        train_well_dfs,
        test_well_df,
        features_for_scaling,
        hyperparams,
        fold_idx,
        cluster_name=None,
        save_to_disk=True,
    ):
        sequence_length = self.base_config["sequence_length"]
        mask_value = self.base_config["mask_value"]
        target_feature = self.base_config.get("target_feature", "VS")

        set_deterministic(self.base_config.get("seed", 42))

        # Fit scaler on training data only (prevents scaler leakage into val)
        combined_train_df = pl.concat(train_well_dfs, how="vertical")
        combined_train_df_cleaned = combined_train_df.drop_nulls(
            subset=[target_feature]
        )
        scaler = RobustScaler()
        if combined_train_df_cleaned.height > 0 and features_for_scaling:
            scaler.fit(
                combined_train_df_cleaned.select(features_for_scaling).to_numpy()
            )

        # Preprocess training wells with the fitted scaler
        all_train_features, all_train_targets = [], []
        well_segments = []
        for df in train_well_dfs:
            processed = self.preprocess_df(
                df, scaler, features_for_scaling, target_feature
            )
            well_segments.append(
                (
                    processed.select(features_to_use).to_numpy(),
                    processed.get_column(target_feature).to_numpy(),
                )
            )
            all_train_features.extend(
                processed.select(features_to_use).to_numpy().tolist()
            )
            all_train_targets.extend(
                processed.get_column(target_feature).to_list()
            )

        # Preprocess test well with the same scaler
        processed_test = self.preprocess_df(
            test_well_df, scaler, features_for_scaling, target_feature
        )
        test_features_data = processed_test.select(features_to_use).to_numpy()
        test_target_data = processed_test.get_column(target_feature).to_numpy()

        augmentation = create_augmentation(self.base_config)
        train_dataset = WellLogDataset(
            None,
            None,
            sequence_length,
            mask_value,
            augmentation=augmentation,
            well_segments=well_segments,
        )
        test_dataset = WellLogDataset(
            test_features_data,
            test_target_data,
            sequence_length,
            mask_value,
            augmentation=None,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
        )

        model = Rebuilt_SAIDNN(
            n_features=len(features_to_use),
            sequence_length=sequence_length,
            embed_dim=hyperparams["embed_dim"],
            num_heads=hyperparams["num_heads"],
            num_blocks=hyperparams["num_blocks"],
            dropout=hyperparams["dropout"],
            use_attention_pooling=True,
        )

        if hyperparams["criterion_type"] == "huber":
            criterion = nn.HuberLoss(delta=1.0)
        elif hyperparams["criterion_type"] == "mse":
            criterion = nn.MSELoss()
        else:  # mae
            criterion = nn.L1Loss()

        weight_decay = hyperparams.get("weight_decay", 0.0)
        if hyperparams["optimizer_type"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams["learning_rate"],
                weight_decay=weight_decay,
            )
        elif hyperparams["optimizer_type"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hyperparams["learning_rate"],
                weight_decay=weight_decay,
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hyperparams["learning_rate"],
                momentum=0.9,
                weight_decay=weight_decay,
            )

        if hyperparams["scheduler_type"] == "plateau":
            base_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
        elif hyperparams["scheduler_type"] == "cosine":
            base_scheduler = CosineAnnealingLR(
                optimizer, T_max=self.base_config["num_epochs"]
            )
        else:  # step
            base_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        scheduler = WarmupScheduler(
            optimizer, warmup_epochs=5, base_scheduler=base_scheduler
        )

        trained_model, history, best_val_loss = train_model_with_validation_split(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            self.base_config["num_epochs"],
            self.base_config["patience"],
            verbose=False,
        )

        test_metrics = evaluate_model(trained_model, test_loader, criterion)

        model_id = (
            f"fold{fold_idx}_features{'_'.join([f[:3] for f in features_to_use])}_"
            f"emb{hyperparams['embed_dim']}_heads{hyperparams['num_heads']}_"
            f"blocks{hyperparams['num_blocks']}"
        )

        if save_to_disk:
            model_path = os.path.join(self.results_dir, f"{model_id}_model.pth")
            scaler_path = os.path.join(self.results_dir, f"{model_id}_scaler.pkl")
            torch.save(trained_model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
        else:
            model_path = ""
            scaler_path = ""

        return {
            "fold_idx": fold_idx,
            "cluster_name": cluster_name,
            "features": features_to_use,
            "hyperparams": hyperparams,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "history": history,
            "final_train_loss": history["train_loss"][-1],
            "final_train_r2": history["train_r2"][-1],
            "model_params": sum(p.numel() for p in trained_model.parameters()),
            "model_id": model_id,
        }

    def _prepare_cluster_folds(
        self, feature_combinations, wells_with_target, clusters, target_feature
    ):
        """Precompute fold arrays once, so Optuna trials only retrain.

        Returns raw well DataFrames per fold.  The scaler is intentionally
        **not** fit here — it will be fit on the inner training split only,
        preventing scaler leakage into the validation set.

        Returns:
 {cluster_name: {features_key: [fold_dict, ...]}}
        """

        fold_data: dict[str, dict[tuple, list]] = {}
        for features_to_use in feature_combinations:
            features_key = tuple(features_to_use)
            for cluster_name, cluster_index in clusters.items():
                folds = []
                for test_well_idx in cluster_index:
                    train_idxs = [i for i in cluster_index if i != test_well_idx]
                    if not train_idxs:
                        continue

                    train_wells_dfs = [wells_with_target[i] for i in train_idxs]
                    test_well_df = wells_with_target[test_well_idx]

                    combined_train_df = pl.concat(train_wells_dfs, how="vertical")
                    combined_train_df_cleaned = combined_train_df.drop_nulls(
                        subset=[target_feature]
                    )

                    if combined_train_df_cleaned.height == 0:
                        continue

                    folds.append(
                        {
                            "train_well_dfs": train_wells_dfs,
                            "test_well_df": test_well_df,
                            "features_for_scaling": [
                                f for f in features_to_use if f != "DEPT"
                            ],
                            "fold_idx": test_well_idx,
                        }
                    )

                if folds:
                    fold_data.setdefault(cluster_name, {})[features_key] = folds
        return fold_data

    def _objective(self, trial, cluster_feature_folds, cluster_name):
        # Feature selection as a categorical search variable.
        combo_keys = list(cluster_feature_folds.keys())
        combo_idx = trial.suggest_categorical(
            "feature_combo_idx", list(range(len(combo_keys)))
        )
        features_key = combo_keys[combo_idx]
        features_to_use = list(features_key)
        folds = cluster_feature_folds[features_key]

        # Optuna needs a static value space per parameter: sample independently
        # and then round embed_dim down to a multiple of num_heads so the
        # MultiheadAttention works for every combination.
        embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 160, 256])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 10])
        embed_dim = max(num_heads, embed_dim - (embed_dim % num_heads))
        hyperparams = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-3, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "scheduler_type": trial.suggest_categorical(
                "scheduler_type", ["plateau", "cosine", "step"]
            ),
            "criterion_type": trial.suggest_categorical(
                "criterion_type", ["huber", "mse", "mae"]
            ),
            "optimizer_type": trial.suggest_categorical(
                "optimizer_type", ["adam", "adamw", "sgd"]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }

        fold_r2 = []
        best_metrics = None
        for fold in folds:
            result = self.run_single_experiment(
                features_to_use,
                fold["train_well_dfs"],
                fold["test_well_df"],
                fold["features_for_scaling"],
                hyperparams,
                fold_idx=fold["fold_idx"],
                cluster_name=cluster_name,
                save_to_disk=False,
            )
            r2 = float(result["test_metrics"]["r2"])
            fold_r2.append(r2)
            if best_metrics is None or r2 >= best_metrics["r2"]:
                best_metrics = {
                    k: float(v) for k, v in result["test_metrics"].items()
                }

            # Prune early trials that clearly lag behind.
            trial.report(float(np.mean(fold_r2)), step=len(fold_r2))
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_r2", fold_r2)
        trial.set_user_attr("best_fold_metrics", best_metrics)
        trial.set_user_attr("hyperparams", hyperparams)
        trial.set_user_attr("features", features_to_use)
        return float(np.mean(fold_r2))

    def run_optuna_optimization(
        self,
        feature_combinations,
        wells_data,
        target_feature="VS",
        n_trials=30,
        n_jobs=10,
        n_startup_trials=3,
    ):
        wells_with_target = [
            df for df in wells_data if target_feature in df.columns
        ]
        clusters = self.base_config.get(
            "clusters", {"A": list(range(len(wells_with_target)))}
        )
        labeled_well_indices = self.base_config.get("labeled_well_indices")
        if labeled_well_indices:
            clusters = normalize_clusters_for_labeled_wells(
                clusters,
                labeled_well_indices,
                len(wells_with_target),
            )

        valid_clusters = {}
        for cluster_name, cluster_indices in clusters.items():
            normalized_indices = sorted(set(int(index) for index in cluster_indices))
            invalid_indices = [
                index
                for index in normalized_indices
                if index < 0 or index >= len(wells_with_target)
            ]
            if invalid_indices:
                raise ValueError(
                    f"Cluster '{cluster_name}' contains invalid well indices "
                    f"{invalid_indices}; {len(wells_with_target)} labeled wells are available."
                )
            if len(normalized_indices) < 2:
                print(
                    f"Skipping cluster '{cluster_name}': at least two labeled wells "
                    "are required for leave-one-well-out validation."
                )
                continue
            valid_clusters[cluster_name] = normalized_indices

        if not valid_clusters:
            raise ValueError(
                "No valid clusters remain for Optuna. Each cluster must contain "
                "at least two selected wells with the target feature."
            )

        fold_data = self._prepare_cluster_folds(
            feature_combinations, wells_with_target, valid_clusters, target_feature
        )

        storage = f"sqlite:///{os.path.join(self.results_dir, 'optuna.db')}"
        best_config = {}

        cluster_count = len(fold_data)
        for cluster_position, (cluster_name, cluster_feature_folds) in enumerate(
            fold_data.items(), start=1
        ):
            if not cluster_feature_folds:
                continue

            total_folds = sum(len(f) for f in cluster_feature_folds.values())
            combo_count = len(cluster_feature_folds)
            study_name = f"cluster_{cluster_name}"

            print(f"\n{'=' * 80}")
            print(f"OPTUNA STUDY: {study_name}")
            print(f"Cluster wells: {clusters[cluster_name]}")
            print(f"Feature combos: {combo_count} | Folds per combo: {list(len(f) for f in cluster_feature_folds.values())}")
            print(f"{'=' * 80}")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=n_startup_trials
                ),
            )
            def on_trial_complete(study, trial):
                if self.progress_callback is None:
                    return
                completed_trials = len(study.trials)
                trial_fraction = min(1.0, completed_trials / max(1, n_trials))
                overall_fraction = (
                    cluster_position - 1 + trial_fraction
                ) / max(1, cluster_count)
                self.progress_callback(
                    15 + overall_fraction * 70,
                    (
                        f"Optuna cluster {cluster_position}/{cluster_count} "
                        f"'{cluster_name}', trial {completed_trials}/{n_trials}"
                    ),
                )

            study.optimize(
                lambda trial: self._objective(
                    trial, cluster_feature_folds, cluster_name
                ),
                n_trials=n_trials,
                n_jobs=n_jobs,
                callbacks=[on_trial_complete],
                show_progress_bar=False,
            )

            completed_trials = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                raise RuntimeError(
                    f"Optuna study '{study_name}' finished without a completed trial. "
                    "Check the preceding trial errors and confirm that each selected "
                    "well has all mapped features and VS values."
                )

            best_trial = study.best_trial
            fold_r2 = best_trial.user_attrs["fold_r2"]
            features_to_use = best_trial.user_attrs.get("features", [])
            effective_hyperparams = best_trial.user_attrs.get(
                "hyperparams", best_trial.params
            )
            print(f"  Best trial #{best_trial.number} | R²: {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
            print(f"  Features: {features_to_use}")
            print(f"  Params: {effective_hyperparams}")

            entry = {
                "cluster": cluster_name,
                "cluster_wells": list(clusters[cluster_name]),
                "features": features_to_use,
                "hyperparams": effective_hyperparams,
                "avg_r2": float(np.mean(fold_r2)),
                "std_r2": float(np.std(fold_r2)),
                "best_fold_r2": float(np.max(fold_r2)),
                "test_metrics": best_trial.user_attrs["best_fold_metrics"],
                "model_path": "",
                "scaler_path": "",
                "model_id": f"optuna_trial_{best_trial.number}",
                "n_trials": len(study.trials),
            }

            best_config[cluster_name] = entry

        return best_config

    def save_best_config(self, best_config):
        best_config_path = os.path.join(self.results_dir, "best_config.json")
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2, default=str)
        print(f"\nBest config saved to {best_config_path}")
        return best_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="Training input JSON payload when using the payload source"
    )
    parser.add_argument(
        "--cluster", help="Recommended clusters JSON path for the payload source"
    )
    parser.add_argument(
        "--output", required=True, help="Directory to store experiment results"
    )
    parser.add_argument(
        "--data-source",
        choices=("payload", "petrobras"),
        default="payload",
        help="Training data source (default: payload)",
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=3)
    default_model_root = Path(__file__).resolve().parent / "data" / "petrobras"
    parser.add_argument(
        "--model_root",
        default=str(default_model_root),
        help="Directory containing the source LAS files",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()

    contract_path = args.contract
    progress_path = args.progress
    operation = "optuna_training"
    write_execution_contract(
        contract_path, args.job_id, operation, "running", "Execution started."
    )
    emit_progress(progress_path, args.job_id, operation, 0, "starting", "Execution started.")

    try:
        emit_progress(
            progress_path, args.job_id, operation, 10, "loading", "Input loading started."
        )
        if args.data_source == "petrobras":
            if args.input or args.cluster:
                parser.error("--input and --cluster are not used with --data-source petrobras")

            feature_combinations = [
                ["VP", "RHO", "GR", "CALIPER", "POROSIDADE", "SATURACAO", "ARGILOSIDADE"],
                ["VP", "RHO", "GR", "CALIPER", "POROSIDADE", "SATURACAO"],
                ["VP", "RHO", "ACOUSTIC_IMP", "GR", "POROSIDADE", "SATURACAO"],
                ["VP", "RHO", "POROSIDADE", "GR", "ARGILOSIDADE"],
                ["VP", "RHO", "POROSIDADE", "SATURACAO"],
            ]
            wells = read_all_wells_with_dept_to_list(features="all")
            wells = filter_commom_features(wells, ignore=["VS"])
            wells = [add_derived_features(df) for df in wells]
            wells_with_vs = [df for df in wells if "VS" in df.columns]
            wells_without_vs = [df for df in wells if "VS" not in df.columns]
            if len(wells_with_vs) != 5:
                raise ValueError(
                    "Expected five Petrobras wells with measured VS (WELL_2-WELL_6); "
                    f"found {len(wells_with_vs)}."
                )
            base_config = {
                "sequence_length": 15,
                "mask_value": -1.0,
                "num_epochs": 500,
                "patience": 150,
                "target_feature": "VS",
                "clusters": {"petrobras_all_measured_vs": list(range(len(wells_with_vs)))},
            }
        else:
            if not args.input or not args.cluster:
                parser.error("--input and --cluster are required with --data-source payload")
            predictor = Predictor(model_root=args.model_root, device=None)
            base_config, feature_combinations, wells_with_vs, wells_without_vs = (
                predictor.prediction_setup(args.input, args.output, args.cluster)
            )

        def report_optuna_progress(percent, message):
            emit_progress(
                progress_path,
                args.job_id,
                operation,
                percent,
                "optimization",
                message,
            )

        experiment = OptunaCrossFoldExperiment(
            base_config,
            results_dir=args.output,
            progress_callback=report_optuna_progress,
        )
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        if args.startup_trials < 0 or args.startup_trials > args.trials:
            parser.error("--startup-trials must be between 0 and --trials")
        if args.jobs < 1:
            parser.error("--jobs must be at least 1")

        best_config = experiment.run_optuna_optimization(
            feature_combinations,
            wells_with_vs,
            n_trials=args.trials,
            n_jobs=args.jobs,
            n_startup_trials=args.startup_trials,
        )
        experiment.save_best_config(best_config)

        emit_progress(
            progress_path,
            args.job_id,
            operation,
            90,
            "finalizing",
            "Final models and reports are being written.",
        )
        final_output_dir = os.path.join(args.output, "final_model")
        trainer = FinalModelTrainer(
            load_best_configuration(args.output), base_config, output_dir=final_output_dir
        )
        trained_clusters = trainer.train_final_model(
            wells_with_vs,
            target_feature=base_config["target_feature"],
            progress_callback=(
                lambda cluster_name, cluster_position, cluster_count, epoch,
                total_epochs, metrics: emit_progress(
                    progress_path,
                    args.job_id,
                    operation,
                    85 + (
                        (
                            cluster_position - 1
                            + epoch / max(1, total_epochs)
                        )
                        / max(1, cluster_count)
                    ) * 10,
                    "final_training",
                    (
                        f"Cluster {cluster_position}/{cluster_count} ({cluster_name}), "
                        f"epoch {epoch}/{total_epochs}; "
                        f"validation loss={metrics['val_loss']:.5f}"
                    ),
                )
            ),
        )
        all_results = trainer.predict_on_wells(
            trained_clusters,
            wells_without_vs,
            wells_with_vs,
            target_feature=base_config["target_feature"],
        )
        trainer.plot_predictions(all_results)
        trainer.generate_summary_report(all_results, load_best_configuration(args.output))
        with open(os.path.join(args.output, "training_result.json"), "w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "schema_version": 1,
                    "job_id": args.job_id,
                    "status": "success",
                    "experiment_dir": os.path.abspath(args.output),
                    "final_model_dir": os.path.abspath(final_output_dir),
                },
                file_handle,
                indent=2,
            )
        emit_progress(
            progress_path, args.job_id, operation, 100, "completed", "Execution completed."
        )
        write_execution_contract(
            contract_path, args.job_id, operation, "success", "Execution completed."
        )
    except KeyboardInterrupt:
        write_progress_contract(
            progress_path, args.job_id, operation, 100, "cancelled", "Execution cancelled."
        )
        write_execution_contract(
            contract_path, args.job_id, operation, "cancelled", "Execution cancelled."
        )
        raise
    except Exception as error:
        write_progress_contract(
            progress_path, args.job_id, operation, 100, "error", str(error)
        )
        write_execution_contract(
            contract_path,
            args.job_id,
            operation,
            "error",
            str(error),
            type(error).__name__,
        )
        raise


if __name__ == "__main__":
    main()
