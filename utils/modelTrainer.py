import json
import os
from datetime import datetime
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from selfAttention import Rebuilt_SAIDNN
from utils.training_utilities import (
    WarmupScheduler,
    calculate_metrics,
    train_model_with_validation_split,
)
from utils.WellLogDataset import WellLogAugmentation, WellLogDataset


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and np.isnan(value)


class FinalModelTrainer:
    """Train final model on all data with best hyperparameters."""

    def __init__(
        self,
        best_config: dict[str, Any],
        base_config: dict[str, Any],
        output_dir: str | None = None,
    ):
        self.base_config = base_config
        self.best_config = best_config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"final_model_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

    def preprocess_df(
        self,
        df: pl.DataFrame,
        scaler: RobustScaler,
        features_for_scaling: list[str],
        target_feature: str,
        mask_value: float = -1.0,
    ) -> pl.DataFrame:
        processed_df = df.clone()

        if len(processed_df) > 0 and features_for_scaling:
            feature_matrix = processed_df.select(features_for_scaling).to_numpy()
            scaled_values = scaler.transform(feature_matrix)
            processed_df = processed_df.with_columns(
                [
                    pl.Series(col, scaled_values[:, index])
                    for index, col in enumerate(features_for_scaling)
                ]
            )

            processed_df = processed_df.with_columns(
                [
                    pl.col(col).fill_nan(mask_value).fill_null(mask_value)
                    for col in features_for_scaling
                ]
            )

        if target_feature in processed_df.columns:
            processed_df = processed_df.with_columns(
                pl.col(target_feature).fill_nan(mask_value).fill_null(mask_value)
            )
        else:
            processed_df = processed_df.with_columns(
                pl.lit(mask_value).alias(target_feature)
            )

        return processed_df

    def train_final_model(
        self, wells_with_vs: list[pl.DataFrame], target_feature: str = "VS"
    ):
        """Train final model on all wells with VS."""
        print(f"\n{'=' * 80}")
        print("TRAINING FINAL MODEL ON ALL WELLS WITH VS")
        print(f"{'=' * 80}")

        features_to_use = self.best_config["features"]
        hyperparams = self.best_config["hyperparams"]

        print(f"Features: {features_to_use}")
        print(f"Hyperparameters: {hyperparams}")

        if not wells_with_vs:
            raise ValueError("wells_with_vs cannot be empty")

        combined_train_df = pl.concat(wells_with_vs, how="vertical_relaxed")
        combined_train_df_cleaned = combined_train_df.filter(
            pl.col(target_feature).is_not_null() & ~pl.col(target_feature).is_nan()
        )

        scaler = RobustScaler()
        features_for_scaling = [
            feature for feature in features_to_use if feature != "DEPT"
        ]
        if features_for_scaling:
            scaler.fit(
                combined_train_df_cleaned.select(features_for_scaling).to_numpy()
            )

        processed_train_dfs = []
        for df in wells_with_vs:
            processed_df = self.preprocess_df(
                df, scaler, features_for_scaling, target_feature
            )
            processed_train_dfs.append(processed_df)

        all_train_features = []
        all_train_targets = []
        for df in processed_train_dfs:
            all_train_features.extend(df.select(features_to_use).to_numpy().tolist())
            all_train_targets.extend(df.select(target_feature).to_series().to_list())

        print(f"Total training samples: {len(all_train_features)}")

        sequence_length = self.base_config["sequence_length"]
        mask_value = self.base_config["mask_value"]

        augmentation = WellLogAugmentation(noise_level=0.01, scale_range=(0.95, 1.05))
        train_dataset = WellLogDataset(
            all_train_features,
            all_train_targets,
            sequence_length,
            mask_value,
            augmentation=augmentation,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
        )

        print(f"Training sequences: {len(train_dataset)}")

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
        else:
            criterion = nn.L1Loss()

        if hyperparams["optimizer_type"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=hyperparams["learning_rate"]
            )
        elif hyperparams["optimizer_type"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=hyperparams["learning_rate"]
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=hyperparams["learning_rate"], momentum=0.9
            )

        if hyperparams["scheduler_type"] == "plateau":
            base_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
        elif hyperparams["scheduler_type"] == "cosine":
            base_scheduler = CosineAnnealingLR(
                optimizer, T_max=self.base_config["num_epochs"]
            )
        else:
            base_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        scheduler = WarmupScheduler(
            optimizer, warmup_epochs=5, base_scheduler=base_scheduler
        )

        print("\nTraining final model...")
        trained_model, history, best_val_loss = train_model_with_validation_split(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            self.base_config["num_epochs"],
            self.base_config["patience"],
            verbose=True,
        )

        print("\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training R²: {history['train_r2'][-1]:.4f}")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")

        model_path = os.path.join(self.output_dir, "final_model.pth")
        scaler_path = os.path.join(self.output_dir, "final_scaler.pkl")

        torch.save(trained_model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as file_handle:
            json.dump(history, file_handle, indent=2)

        self.plot_training_history(history)

        return trained_model, scaler, features_to_use

    def plot_training_history(self, history: dict[str, list[float]]) -> None:
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history["train_loss"], label="Training Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(
            "Training and Validation Loss", fontsize=14, fontweight="bold"
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["train_r2"], label="Training R²", linewidth=2)
        axes[1].plot(history["val_r2"], label="Validation R²", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("R²", fontsize=12)
        axes[1].set_title("Training and Validation R²", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "plots", "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {plot_path}")
        plt.close()

    def predict_on_wells(
        self,
        model,
        scaler,
        features_to_use,
        wells_without_vs,
        wells_with_vs,
        target_feature="VS",
    ):
        """Make predictions on wells."""
        sequence_length = self.base_config["sequence_length"]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            model.to(device)
        model.eval()

        features_for_scaling = [
            feature for feature in features_to_use if feature != "DEPT"
        ]
        all_results = {}

        print(f"\n{'=' * 80}")
        print("MAKING PREDICTIONS")
        print(f"{'=' * 80}")

        print(f"\nPredicting on {len(wells_without_vs)} wells WITHOUT VS:")
        for well_idx, well_df in enumerate(wells_without_vs):
            print(f"\n  Well {well_idx + 1}/{len(wells_without_vs)}")

            processed_df = self.preprocess_df(
                well_df, scaler, features_for_scaling, target_feature
            )
            well_features = processed_df.select(features_to_use).to_numpy()

            predictions = []
            depths = []

            for i in range(len(well_features) - sequence_length):
                seq_features = well_features[i : i + sequence_length]
                input_tensor = (
                    torch.tensor(seq_features, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    output = model(input_tensor)
                    predictions.append(output.item())

                if "DEPT" in well_df.columns:
                    row_data = well_df.row(i + sequence_length, named=True)
                    depths.append(row_data["DEPT"])

            all_results[f"well_without_vs_{well_idx}"] = {
                "predictions": predictions,
                "depths": depths,
                "has_ground_truth": False,
            }

            print(f"    Generated {len(predictions)} predictions")
            if predictions:
                print(
                    f"    VS range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]"
                )

        print(f"\nValidating on {len(wells_with_vs)} wells WITH VS:")
        for well_idx, well_df in enumerate(wells_with_vs):
            print(f"\n  Well {well_idx + 1}/{len(wells_with_vs)}")

            processed_df = self.preprocess_df(
                well_df, scaler, features_for_scaling, target_feature
            )
            well_features = processed_df.select(features_to_use).to_numpy()

            predictions = []
            actuals = []
            depths = []

            for i in range(len(well_features) - sequence_length):
                seq_features = well_features[i : i + sequence_length]
                input_tensor = (
                    torch.tensor(seq_features, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    output = model(input_tensor)
                    predictions.append(output.item())

                row_data = well_df.row(i + sequence_length, named=True)
                actual_val = row_data.get(target_feature)
                actuals.append(
                    actual_val if not _is_missing_value(actual_val) else np.nan
                )

                if "DEPT" in well_df.columns:
                    depths.append(row_data["DEPT"])

            metrics = calculate_metrics(predictions, actuals)

            all_results[f"well_with_vs_{well_idx}"] = {
                "predictions": predictions,
                "actuals": actuals,
                "depths": depths,
                "metrics": metrics,
                "has_ground_truth": True,
            }

            print(f"    Generated {len(predictions)} predictions")
            print(f"    R²: {metrics['R2']:.4f}")
            print(f"    RMSE: {metrics['RMSE']:.4f}")
            print(f"    MSE: {metrics['MSE']:.4f}")
            print(f"    MAE: {metrics['MAE']:.4f}")

        predictions_path = os.path.join(self.output_dir, "all_predictions.json")

        serializable_results = {}
        for key, value in all_results.items():
            serializable_results[key] = {
                field: field_value
                if not isinstance(field_value, np.ndarray)
                else field_value.tolist()
                for field, field_value in value.items()
            }

        with open(predictions_path, "w", encoding="utf-8") as file_handle:
            json.dump(serializable_results, file_handle, indent=2, default=str)

        print(f"\nAll predictions saved to: {predictions_path}")

        return all_results

    def plot_predictions(self, all_results):
        """Generate comprehensive prediction plots."""
        print(f"\n{'=' * 80}")
        print("GENERATING PLOTS")
        print(f"{'=' * 80}")

        wells_without_vs = {
            key: value
            for key, value in all_results.items()
            if not value["has_ground_truth"]
        }

        if wells_without_vs:
            n_wells = len(wells_without_vs)
            fig, axes = plt.subplots(n_wells, 1, figsize=(12, 4 * n_wells))
            if n_wells == 1:
                axes = [axes]

            for idx, (well_name, data) in enumerate(wells_without_vs.items()):
                ax = axes[idx]

                if data["depths"]:
                    ax.plot(
                        data["depths"],
                        data["predictions"],
                        "b-",
                        linewidth=1.5,
                        label="Predicted VS",
                    )
                    ax.set_xlabel("Depth (m)", fontsize=11)
                else:
                    ax.plot(
                        data["predictions"], "b-", linewidth=1.5, label="Predicted VS"
                    )
                    ax.set_xlabel("Sample Index", fontsize=11)

                ax.set_ylabel("VS", fontsize=11)
                ax.set_title(
                    f"{well_name.replace('_', ' ').title()} - VS Prediction",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()

            plt.tight_layout()
            plot_path = os.path.join(
                self.output_dir, "plots", "predictions_wells_without_vs.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Predictions plot (wells without VS) saved to: {plot_path}")
            plt.close()

        wells_with_vs = {
            key: value
            for key, value in all_results.items()
            if value["has_ground_truth"]
        }

        if wells_with_vs:
            n_wells = len(wells_with_vs)

            fig, axes = plt.subplots(n_wells, 1, figsize=(14, 4 * n_wells))
            if n_wells == 1:
                axes = [axes]

            for idx, (well_name, data) in enumerate(wells_with_vs.items()):
                ax = axes[idx]

                mask = ~np.isnan(data["actuals"])
                actuals_clean = np.array(data["actuals"])[mask]
                predictions_clean = np.array(data["predictions"])[mask]

                if data["depths"]:
                    depths_clean = np.array(data["depths"])[mask]
                    ax.plot(
                        depths_clean,
                        actuals_clean,
                        "g-",
                        linewidth=1.5,
                        label="Actual VS",
                        alpha=0.7,
                    )
                    ax.plot(
                        depths_clean,
                        predictions_clean,
                        "b--",
                        linewidth=1.5,
                        label="Predicted VS",
                        alpha=0.7,
                    )
                    ax.set_xlabel("Depth (m)", fontsize=11)
                else:
                    indices = np.arange(len(actuals_clean))
                    ax.plot(
                        indices,
                        actuals_clean,
                        "g-",
                        linewidth=1.5,
                        label="Actual VS",
                        alpha=0.7,
                    )
                    ax.plot(
                        indices,
                        predictions_clean,
                        "b--",
                        linewidth=1.5,
                        label="Predicted VS",
                        alpha=0.7,
                    )
                    ax.set_xlabel("Sample Index", fontsize=11)

                ax.set_ylabel("VS", fontsize=11)

                metrics = data["metrics"]
                title = (
                    f"{well_name.replace('_', ' ').title()}\n"
                    f"R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                    f"MSE={metrics['MSE']:.3f}, MAE={metrics['MAE']:.3f}"
                )
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()

            plt.tight_layout()
            plot_path = os.path.join(
                self.output_dir, "plots", "predictions_vs_actuals_timeseries.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Time series comparison plot saved to: {plot_path}")
            plt.close()

            fig, axes = plt.subplots(1, n_wells, figsize=(5 * n_wells, 5))
            if n_wells == 1:
                axes = [axes]

            for idx, (well_name, data) in enumerate(wells_with_vs.items()):
                ax = axes[idx]

                mask = ~np.isnan(data["actuals"])
                actuals_clean = np.array(data["actuals"])[mask]
                predictions_clean = np.array(data["predictions"])[mask]

                ax.scatter(
                    actuals_clean,
                    predictions_clean,
                    alpha=0.5,
                    s=20,
                    edgecolors="k",
                    linewidth=0.5,
                )

                min_val = min(actuals_clean.min(), predictions_clean.min())
                max_val = max(actuals_clean.max(), predictions_clean.max())
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    linewidth=2,
                    label="Perfect Prediction",
                )

                ax.set_xlabel("Actual VS", fontsize=11)
                ax.set_ylabel("Predicted VS", fontsize=11)

                metrics = data["metrics"]
                title = f"{well_name.replace('_', ' ').title()}\nR²={metrics['R2']:.3f}"
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plot_path = os.path.join(
                self.output_dir, "plots", "predictions_vs_actuals_scatter.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Scatter plot saved to: {plot_path}")
            plt.close()

            fig, axes = plt.subplots(1, n_wells, figsize=(5 * n_wells, 5))
            if n_wells == 1:
                axes = [axes]

            for idx, (well_name, data) in enumerate(wells_with_vs.items()):
                ax = axes[idx]

                mask = ~np.isnan(data["actuals"])
                actuals_clean = np.array(data["actuals"])[mask]
                predictions_clean = np.array(data["predictions"])[mask]
                errors = predictions_clean - actuals_clean

                ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
                ax.axvline(
                    0, color="r", linestyle="--", linewidth=2, label="Zero Error"
                )
                ax.axvline(
                    np.mean(errors),
                    color="g",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean Error: {np.mean(errors):.3f}",
                )

                ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=11)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.set_title(
                    f"{well_name.replace('_', ' ').title()}\nError Distribution",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "plots", "error_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Error distribution plot saved to: {plot_path}")
            plt.close()

    def generate_summary_report(self, all_results, best_config):
        """Generate comprehensive summary report."""
        report_path = os.path.join(self.output_dir, "SUMMARY_REPORT.txt")

        with open(report_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("=" * 80 + "\n")
            file_handle.write("WELL LOG VS PREDICTION - FINAL MODEL SUMMARY REPORT\n")
            file_handle.write("=" * 80 + "\n\n")

            file_handle.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            file_handle.write("-" * 80 + "\n")
            file_handle.write("BEST MODEL CONFIGURATION\n")
            file_handle.write("-" * 80 + "\n")
            file_handle.write(f"Features: {best_config['features']}\n")
            file_handle.write("Hyperparameters:\n")
            for key, value in best_config["hyperparams"].items():
                file_handle.write(f"  {key}: {value}\n")
            file_handle.write("\nAverage Cross-Validation Performance:\n")
            file_handle.write(f"  R²: {best_config.get('avg_test_r2', 'N/A')}\n")
            file_handle.write(f"  RMSE: {best_config.get('avg_test_rmse', 'N/A')}\n")
            file_handle.write(f"  MSE: {best_config.get('avg_test_mse', 'N/A')}\n")
            file_handle.write(f"  MAE: {best_config.get('avg_test_mae', 'N/A')}\n\n")

            wells_without_vs = {
                key: value
                for key, value in all_results.items()
                if not value["has_ground_truth"]
            }
            if wells_without_vs:
                file_handle.write("-" * 80 + "\n")
                file_handle.write("PREDICTIONS ON WELLS WITHOUT VS (NO GROUND TRUTH)\n")
                file_handle.write("-" * 80 + "\n")
                for well_name, data in wells_without_vs.items():
                    file_handle.write(f"\n{well_name}:\n")
                    file_handle.write(
                        f"  Number of predictions: {len(data['predictions'])}\n"
                    )
                    if data["predictions"]:
                        file_handle.write(
                            f"  VS range: [{np.min(data['predictions']):.3f}, "
                            f"{np.max(data['predictions']):.3f}]\n"
                        )
                        file_handle.write(
                            f"  VS mean: {np.mean(data['predictions']):.3f}\n"
                        )
                        file_handle.write(
                            f"  VS std: {np.std(data['predictions']):.3f}\n"
                        )
                    else:
                        file_handle.write("  No predictions generated\n")
                file_handle.write("\n")

            wells_with_vs = {
                key: value
                for key, value in all_results.items()
                if value["has_ground_truth"]
            }
            if wells_with_vs:
                file_handle.write("-" * 80 + "\n")
                file_handle.write("VALIDATION ON WELLS WITH VS (HAS GROUND TRUTH)\n")
                file_handle.write("-" * 80 + "\n")

                all_metrics = []
                for well_name, data in wells_with_vs.items():
                    metrics = data["metrics"]
                    all_metrics.append(metrics)

                    file_handle.write(f"\n{well_name}:\n")
                    file_handle.write(
                        f"  Number of predictions: {len(data['predictions'])}\n"
                    )
                    file_handle.write(f"  R²: {metrics['R2']:.4f}\n")
                    file_handle.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                    file_handle.write(f"  MSE: {metrics['MSE']:.4f}\n")
                    file_handle.write(f"  MAE: {metrics['MAE']:.4f}\n")

                file_handle.write("\nAVERAGE METRICS ACROSS ALL VALIDATION WELLS:\n")
                avg_r2 = np.mean([metrics["R2"] for metrics in all_metrics])
                avg_rmse = np.mean([metrics["RMSE"] for metrics in all_metrics])
                avg_mse = np.mean([metrics["MSE"] for metrics in all_metrics])
                avg_mae = np.mean([metrics["MAE"] for metrics in all_metrics])

                file_handle.write(
                    f"  R²: {avg_r2:.4f} ± {np.std([metrics['R2'] for metrics in all_metrics]):.4f}\n"
                )
                file_handle.write(
                    f"  RMSE: {avg_rmse:.4f} ± {np.std([metrics['RMSE'] for metrics in all_metrics]):.4f}\n"
                )
                file_handle.write(
                    f"  MSE: {avg_mse:.4f} ± {np.std([metrics['MSE'] for metrics in all_metrics]):.4f}\n"
                )
                file_handle.write(
                    f"  MAE: {avg_mae:.4f} ± {np.std([metrics['MAE'] for metrics in all_metrics]):.4f}\n"
                )
                file_handle.write("\n")

            file_handle.write("=" * 80 + "\n")
            file_handle.write("END OF REPORT\n")
            file_handle.write("=" * 80 + "\n")

        print(f"\nSummary report saved to: {report_path}")

        with open(report_path, "r", encoding="utf-8") as file_handle:
            print(f"\n{file_handle.read()}")


def load_best_configuration(experiment_dir: str):
    """Load best configuration from cross-validation."""
    # Handle relative paths - if only the experiment name is provided, prepend 'experiments/'
    if not experiment_dir.startswith("experiments/") and os.path.exists(
        os.path.join("experiments", experiment_dir)
    ):
        experiment_dir = os.path.join("experiments", experiment_dir)

    averaged_results_path = os.path.join(experiment_dir, "averaged_results.json")

    if not os.path.exists(averaged_results_path):
        raise FileNotFoundError(f"Could not find {averaged_results_path}")

    with open(averaged_results_path, "r", encoding="utf-8") as file_handle:
        averaged_results = json.load(file_handle)

    best_config = averaged_results[0]

    print(f"\n{'=' * 80}")
    print("LOADED BEST CONFIGURATION FROM CROSS-VALIDATION")
    print(f"{'=' * 80}")
    print(f"Features: {best_config['features']}")
    print(
        f"Average Test R²: {best_config['avg_test_r2']:.4f} ± {best_config['std_test_r2']:.4f}"
    )
    print(
        f"Average Test RMSE: {best_config['avg_test_rmse']:.4f} ± {best_config['std_test_rmse']:.4f}"
    )

    return best_config
