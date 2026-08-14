import json
import os
from datetime import datetime
from typing import Any, TypedDict

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from selfAttention import Rebuilt_SAIDNN
from utils.modelTrainer import (
    train_model_with_validation_split,
)
from utils.training_utilities import (
    WarmupScheduler,
    evaluate_model,
    set_deterministic,
)
from utils.WellLogDataset import WellLogAugmentation, WellLogDataset

from dataclasses import dataclass


@dataclass
class TestMetrics(TypedDict, total=False):
    R2: float
    RMSE: float
    MSE: float
    MAE: float
    r2: float
    rmse: float
    mse: float
    mae: float
    loss: float


class ClusterBestConfig(TypedDict):
    cluster: str
    cluster_wells: list[int]
    features: list[str]
    hyperparams: dict[str, float]
    avg_r2: float
    std_r2: float
    best_fold_r2: float
    test_metrics: TestMetrics
    model_path: str
    scaler_path: str
    model_id: str


class CrossFoldHyperparameterExperiment:
    def __init__(
        self, base_config: dict[str, Any], results_dir: str | None = None
    ):
        self.base_config: dict[str, Any] = base_config
        self.result = {}
        self.experiment_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_dir is None:
            results_dir = os.path.join("results", self.experiment_id)
        self.results_dir: str = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def define_hyperparameter_configurations(self):
        """Define LARGER model configurations"""
        configurations = [
            # WIDER MODELS
            {
                "embed_dim": 160,
                "num_heads": 10,  # More heads
                "num_blocks": 3,
                "dropout": 0.2,
                "learning_rate": 0.0005,
                "batch_size": 32,
                "scheduler_type": "plateau",
                "criterion_type": "huber",
                "optimizer_type": "adamw",
            },
        ]

        return configurations

    def run_single_experiment(
        self,
        features_to_use,
        all_train_features,
        all_train_targets,
        test_features_data,
        test_target_data,
        scaler,
        hyperparams,
        fold_idx,
        cluster_name=None,
    ):
        sequence_length = self.base_config["sequence_length"]
        mask_value = self.base_config["mask_value"]

        set_deterministic(self.base_config.get("seed", 42))

        # Create augmentation
        augmentation = WellLogAugmentation(noise_level=0.01, scale_range=(0.95, 1.05))

        # Create datasets
        train_dataset = WellLogDataset(
            all_train_features,
            all_train_targets,
            sequence_length,
            mask_value,
            augmentation=augmentation,
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

        # Create model (using SAIDNN_v3)
        model = Rebuilt_SAIDNN(
            n_features=len(features_to_use),
            sequence_length=sequence_length,
            embed_dim=hyperparams["embed_dim"],
            num_heads=hyperparams["num_heads"],
            num_blocks=hyperparams["num_blocks"],
            dropout=hyperparams["dropout"],
            use_attention_pooling=True,
        )

        # Create criterion
        if hyperparams["criterion_type"] == "huber":
            criterion = nn.HuberLoss(delta=1.0)
        elif hyperparams["criterion_type"] == "mse":
            criterion = nn.MSELoss()
        else:  # mae
            criterion = nn.L1Loss()

        # Create optimizer
        if hyperparams["optimizer_type"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=hyperparams["learning_rate"]
            )
        elif hyperparams["optimizer_type"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=hyperparams["learning_rate"]
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                model.parameters(), lr=hyperparams["learning_rate"], momentum=0.9
            )

        # Create scheduler
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

        # Wrap with warmup
        scheduler = WarmupScheduler(
            optimizer, warmup_epochs=5, base_scheduler=base_scheduler
        )

        # Train model
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

        # Evaluate
        test_metrics = evaluate_model(trained_model, test_loader, criterion)

        # Create unique identifier
        model_id = (
            f"fold{fold_idx}_features{'_'.join([f[:3] for f in features_to_use])}_"
            f"emb{hyperparams['embed_dim']}_heads{hyperparams['num_heads']}_"
            f"blocks{hyperparams['num_blocks']}"
        )

        model_path = os.path.join(self.results_dir, f"{model_id}_model.pth")
        scaler_path = os.path.join(self.results_dir, f"{model_id}_scaler.pkl")

        torch.save(trained_model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        result = {
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

        return result

    def save_results(self):
        results_path = os.path.join(self.results_dir, "experiment_results.json")

        serializable_results = {}
        for key, value in self.results.items():
            # key agora tem 4 elementos: features, hyperparams, fold_idx, cluster_name
            str_key = f"cluster_{key[3]}_features_{hash(key[0])}_fold_{key[2]}"
            serializable_value = {k: v for k, v in value.items() if k != "history"}
            serializable_value["experiment_id"] = self.experiment_id
            serializable_results[str_key] = serializable_value

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\nResults saved to {results_path}")

    def analyze_results(self):
        if not self.results:
            print("No results to analyze")
            return

        print(f"\n{'=' * 80}")
        print("EXPERIMENT ANALYSIS")
        print(f"{'=' * 80}")

        # Aggregate results across folds
        config_results = {}
        for key, result in self.results.items():
            features = key[0]
            hyperparams = key[1]
            config_key = (features, hyperparams)

            if config_key not in config_results:
                config_results[config_key] = {
                    "test_r2": [],
                    "test_rmse": [],
                    "test_mse": [],
                    "test_mae": [],
                    "test_loss": [],
                }

            config_results[config_key]["test_r2"].append(result["test_metrics"]["r2"])
            config_results[config_key]["test_rmse"].append(
                result["test_metrics"]["rmse"]
            )
            config_results[config_key]["test_mse"].append(result["test_metrics"]["mse"])
            config_results[config_key]["test_mae"].append(result["test_metrics"]["mae"])
            config_results[config_key]["test_loss"].append(
                result["test_metrics"]["loss"]
            )

        # Calculate average metrics
        avg_results = []
        for config_key, metrics in config_results.items():
            avg_result = {
                "features": list(config_key[0]),
                "hyperparams": dict(config_key[1]),
                "avg_test_r2": np.mean(metrics["test_r2"]),
                "std_test_r2": np.std(metrics["test_r2"]),
                "avg_test_rmse": np.mean(metrics["test_rmse"]),
                "std_test_rmse": np.std(metrics["test_rmse"]),
                "avg_test_mse": np.mean(metrics["test_mse"]),
                "std_test_mse": np.std(metrics["test_mse"]),
                "avg_test_mae": np.mean(metrics["test_mae"]),
                "std_test_mae": np.std(metrics["test_mae"]),
            }
            avg_results.append(avg_result)

        # Sort by R²
        avg_results.sort(key=lambda x: x["avg_test_r2"], reverse=True)

        print(f"\n{'=' * 80}")
        print("TOP 10 CONFIGURATIONS BY AVERAGE TEST R²")
        print(f"{'=' * 80}")

        for i, result in enumerate(avg_results[:10]):
            print(f"\n{i + 1}. Configuration:")
            print(f"   Features: {result['features']}")
            print("   Hyperparameters:")
            for key, value in result["hyperparams"].items():
                print(f"     {key}: {value}")
            print(
                f"   Average Test R²: {result['avg_test_r2']:.4f} ± {result['std_test_r2']:.4f}"
            )
            print(
                f"   Average Test RMSE: {result['avg_test_rmse']:.4f} ± {result['std_test_rmse']:.4f}"
            )
            print(
                f"   Average Test MSE: {result['avg_test_mse']:.4f} ± {result['std_test_mse']:.4f}"
            )
            print(
                f"   Average Test MAE: {result['avg_test_mae']:.4f} ± {result['std_test_mae']:.4f}"
            )

        # Save averaged results
        avg_results_path = os.path.join(self.results_dir, "averaged_results.json")
        with open(avg_results_path, "w") as f:
            json.dump(avg_results, f, indent=2, default=str)

        print(f"\nAveraged results saved to {avg_results_path}")

    def save_best_models(self):
        """Salva a melhor configuração por cluster"""
        if not self.results:
            print("No results to save")
            return

        clusters = self.base_config.get("clusters", {"default": []})
        best_per_cluster: Dict[str, ClusterBestConfig] = {}

        for cluster_name in clusters:
            # Filtrar apenas resultados deste cluster
            # A chave é: (tuple(features), tuple(hyperparams.items()), fold_idx, cluster_name)
            cluster_results = {
                key: result
                for key, result in self.results.items()
                if key[3] == cluster_name
            }

            if not cluster_results:
                print(f"  No results found for cluster {cluster_name}, skipping")
                continue

            # Agrupar por configuração (features + hyperparams) e calcular R² médio
            config_scores = {}
            for key, result in cluster_results.items():
                config_key = (key[0], key[1])  # (features, hyperparams)
                if config_key not in config_scores:
                    config_scores[config_key] = []
                config_scores[config_key].append(result["test_metrics"]["r2"])

            # Melhor configuração = maior R² médio dentro do cluster
            best_config_key = max(
                config_scores, key=lambda k: np.mean(config_scores[k])
            )

            # Pegar o resultado com o melhor fold para esse config
            best_result = max(
                [
                    r
                    for k, r in cluster_results.items()
                    if (k[0], k[1]) == best_config_key
                ],
                key=lambda r: r["test_metrics"]["r2"],
            )

            best_per_cluster[cluster_name] = {
                "cluster": cluster_name,
                "cluster_wells": list(clusters[cluster_name]),
                "features": list(best_result["features"]),
                "hyperparams": dict(best_result["hyperparams"]),
                "avg_r2": float(np.mean(config_scores[best_config_key])),
                "std_r2": float(np.std(config_scores[best_config_key])),
                "best_fold_r2": float(best_result["test_metrics"]["r2"]),
                "test_metrics": best_result["test_metrics"],
                "model_path": str(best_result["model_path"]),
                "scaler_path": str(best_result["scaler_path"]),
                "model_id": str(best_result["model_id"]),
            }

        # Salvar JSON
        best_models_path = os.path.join(self.results_dir, "best_config.json")
        with open(best_models_path, "w") as f:
            json.dump(best_per_cluster, f, indent=2, default=str)

        # Print resumo
        print(f"\n{'=' * 80}")
        print("BEST MODEL PER CLUSTER")
        print(f"{'=' * 80}")

        for cluster_name, info in best_per_cluster.items():
            print(f"\nCluster {cluster_name} (Wells {info['cluster_wells']}):")
            print(f"  Features: {info['features']}")
            print(f"  Average R²: {info['avg_r2']:.4f} ± {info['std_r2']:.4f}")
            print(f"  Best fold R²: {info['best_fold_r2']:.4f}")
            print(f"  RMSE: {info['test_metrics']['rmse']:.4f}")
            print(f"  Model path: {info['model_path']}")

        print(f"\nBest configs saved to {best_models_path}")
        return best_per_cluster

    def run_cross_fold_experiments(
        self, feature_combinations, wells_data, target_feature="VS"
    ):
        """Run cross-fold validation using Polars dataframes."""
        hyperparameter_configs = self.define_hyperparameter_configurations()

        if not hasattr(self, "results"):
            self.results = {}

        wells_with_target = [df for df in wells_data if target_feature in df.columns]

        print(f"Total wells with {target_feature}: {len(wells_with_target)}")
        print(f"Running {len(hyperparameter_configs)} hyperparameter configurations")
        print(f"Testing {len(feature_combinations)} feature combinations")
        print(f"Cross-fold validation with {len(wells_with_target)} wells")
        print(
            f"Total experiments: {len(hyperparameter_configs) * len(feature_combinations) * len(wells_with_target)}"
        )

        experiment_count = 0
        total_experiments = (
            len(hyperparameter_configs)
            * len(feature_combinations)
            * len(wells_with_target)
        )

        clusters = self.base_config.get(
            "clusters", {"A": list(range(len(wells_with_target)))}
        )

        for features_to_use in feature_combinations:
            print(f"\n{'=' * 80}")
            print(f"FEATURE COMBINATION: {features_to_use}")
            print(f"{'=' * 80}")

            for cluster_name, cluster_index in clusters.items():
                print(f"\n{'-' * 80}")
                print(f"CLUSTER {cluster_name}: Wells {cluster_index}")
                print(f"{'-' * 80}")
                # cluster_wells = [wells_with_target[i] for i in cluster_index]

                for fold_idx, test_well_idx in enumerate(cluster_index):
                    train_idxs = [i for i in cluster_index if i != test_well_idx]

                    if len(train_idxs) == 0:
                        print(
                            f"  Skipping fold {fold_idx} for cluster {cluster_name} because there are no training wells"
                        )
                        continue

                    print(f"\n{'-' * 80}")
                    print(
                        f"FOLD {fold_idx + 1}/{len(cluster_index)}: Using Well {test_well_idx} as test set"
                    )
                    print(f"{'-' * 80}")

                    train_wells_dfs = [wells_with_target[i] for i in train_idxs]
                    test_well_df = wells_with_target[test_well_idx]

                    combined_train_df = pl.concat(train_wells_dfs, how="vertical")
                    combined_train_df_cleaned = combined_train_df.drop_nulls(
                        subset=[target_feature]
                    )

                    scaler = RobustScaler()
                    features_for_scaling = [f for f in features_to_use if f != "DEPT"]

                    if combined_train_df_cleaned.height == 0:
                        print(
                            f"  No valid training data for features {features_to_use}, fold {test_well_idx}"
                        )
                        continue

                    scaler.fit(
                        combined_train_df_cleaned.select(
                            features_for_scaling
                        ).to_numpy()
                    )

                    processed_train_dfs = []
                    for df in train_wells_dfs:
                        processed_df = self.preprocess_df(
                            df, scaler, features_for_scaling, target_feature
                        )
                        processed_train_dfs.append(processed_df)

                    all_train_features = []
                    all_train_targets = []
                    for df in processed_train_dfs:
                        all_train_features.extend(
                            df.select(features_to_use).to_numpy().tolist()
                        )
                        all_train_targets.extend(
                            df.get_column(target_feature).to_list()
                        )

                    processed_test_df = self.preprocess_df(
                        test_well_df, scaler, features_for_scaling, target_feature
                    )
                    test_features_data = processed_test_df.select(
                        features_to_use
                    ).to_numpy()
                    test_target_data = processed_test_df.get_column(
                        target_feature
                    ).to_numpy()

                    if len(all_train_features) == 0:
                        print(f"  No valid training samples for fold {test_well_idx}")
                        continue

                    for hyperparam_config in hyperparameter_configs:
                        experiment_count += 1
                        print(f"\nExperiment {experiment_count}/{total_experiments}")
                        print(f"  Hyperparameters: {hyperparam_config}")

                        try:
                            result = self.run_single_experiment(
                                features_to_use,
                                all_train_features,
                                all_train_targets,
                                test_features_data,
                                test_target_data,
                                scaler,
                                hyperparam_config,
                                fold_idx=test_well_idx,
                                cluster_name=cluster_name,
                            )

                            experiment_key = (
                                tuple(features_to_use),
                                tuple(hyperparam_config.items()),
                                test_well_idx,
                                cluster_name,
                            )
                            self.results[experiment_key] = result

                            print("  Results:")
                            print(
                                f"    Test Loss: {result['test_metrics']['loss']:.6f}"
                            )
                            print(f"    Test R²: {result['test_metrics']['r2']:.4f}")
                            print(
                                f"    Test RMSE: {result['test_metrics']['rmse']:.4f}"
                            )
                            print(f"    Test MSE: {result['test_metrics']['mse']:.4f}")
                            print(f"    Test MAE: {result['test_metrics']['mae']:.4f}")

                            if experiment_count % 5 == 0:
                                self.save_results()

                        except Exception as e:
                            print(f"  Experiment failed: {e}")
                            import traceback

                            traceback.print_exc()
                            continue

        self.save_results()
        self.analyze_results()
        self.save_best_models()
        return self.results

    def preprocess_df(
        self, df, scaler, features_for_scaling, target_feature, mask_value=-1.0
    ):
        processed_df = df.clone()

        if features_for_scaling and processed_df.height > 0:
            scaled_values = scaler.transform(
                processed_df.select(features_for_scaling).to_numpy()
            )
            processed_df = processed_df.with_columns(
                [
                    pl.Series(name=col, values=scaled_values[:, idx])
                    for idx, col in enumerate(features_for_scaling)
                ]
            )

        for col in features_for_scaling:
            if col in processed_df.columns:
                processed_df = processed_df.with_columns(
                    pl.col(col).fill_null(mask_value).alias(col)
                )

        if target_feature in processed_df.columns:
            processed_df = processed_df.with_columns(
                pl.col(target_feature).fill_null(mask_value).alias(target_feature)
            )
        else:
            processed_df = processed_df.with_columns(
                pl.lit(mask_value).alias(target_feature)
            )

        return processed_df
