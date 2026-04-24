import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import copy
import polars as pl


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add physically meaningful derived features using Polars."""
    derived_df = df.clone()
    expressions: List[pl.Expr] = []

    # Acoustic impedance (VP * RHO)
    if "VP" in derived_df.columns and "RHO" in derived_df.columns:
        expressions.append((pl.col("VP") * pl.col("RHO")).alias("ACOUSTIC_IMP"))

    # VP normalized
    if "VP" in derived_df.columns:
        vp_mean = derived_df.get_column("VP").mean()
        if vp_mean is not None:
            vp_mean_value = float(vp_mean)
            if vp_mean_value > 0:
                expressions.append(
                    (pl.col("VP") / pl.lit(vp_mean_value)).alias("VP_NORMALIZED")
                )

    # Rolling statistics (local context)
    for col in ["VP", "RHO", "GR"]:
        if col in derived_df.columns:
            expressions.append(
                pl.col(col)
                .rolling_mean(window_size=5, center=True)
                .alias(f"{col}_ROLL_MEAN")
            )
            expressions.append(
                pl.col(col)
                .rolling_std(window_size=5, center=True)
                .alias(f"{col}_ROLL_STD")
            )

    if expressions:
        derived_df = derived_df.with_columns(expressions)

    # Fill nulls introduced by rolling operations.
    derived_df = derived_df.fill_null(strategy="backward").fill_null(strategy="forward")

    return derived_df


class WarmupScheduler:
    """Learning rate warmup for stable training"""

    def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
        self.optimizar = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, metrics=None):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase: linearly increase LR
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizar.param_groups:
                param_group["lr"] = lr
        elif self.base_scheduler:
            # After warmup, use the base scheduler if provided
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()


def calculate_metrics(predictions, targets):
    """Calculates R-squared metric, RMSE, MSE and MAE"""
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]

    if len(predictions) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MSE": np.nan, "MAE": np.nan}

    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)

    try:
        r2 = r2_score(targets, predictions)
    except:
        r2 = np.nan

    metrics = {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
    }
    metrics["r2"] = metrics["R2"]
    metrics["rmse"] = metrics["RMSE"]
    metrics["mse"] = metrics["MSE"]
    metrics["mae"] = metrics["MAE"]
    return metrics


def get_predictions_and_targets(model, data_loader, device):
    """Get all predictions and targets from a data loader"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    return all_predictions, all_targets


def train_model_with_validation_split(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    patience,
    val_split=0.2,
    verbose=True,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Split training data for validation
    train_data = list(train_loader.dataset)
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size

    train_subset = torch.utils.data.Subset(train_loader.dataset, range(train_size))
    val_subset = torch.utils.data.Subset(
        train_loader.dataset, range(train_size, len(train_data))
    )

    train_loader_split = DataLoader(
        train_subset, batch_size=train_loader.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=train_loader.batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_train_predictions = []
        epoch_train_targets = []
        for features, targets in train_loader_split:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()

            # Mixed precision foward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_predictions.extend(outputs.detach().cpu().numpy().flatten())
            epoch_train_targets.extend(targets.detach().cpu().numpy().flatten())

            total_loss += loss.item() * features.size(0)
        avg_train_loss = total_loss / len(train_loader_split)
        history["train_loss"].append(avg_train_loss)
        train_metrics = calculate_metrics(epoch_train_predictions, epoch_train_targets)
        history["train_r2"].append(train_metrics["R2"])

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(features)
                        val_loss = criterion(outputs, targets)
                else:
                    outputs = model(features)
                    val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Calculate validation metrics
        val_predictions, val_targets = get_predictions_and_targets(
            model, val_loader, device
        )
        val_metrics = calculate_metrics(val_predictions, val_targets)
        history["val_r2"].append(val_metrics["R2"])

        if hasattr(scheduler, "step"):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, WarmupScheduler):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} | Val R2: {val_metrics['R2']:.4f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, history, best_val_loss


def evaluate_model(model, test_loader, criterion):
    """
    Evaluete model and return loss and metrics
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    use_amp = torch.cuda.is_available()

    total_test_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)

    # Calculate metrics
    test_predictions, test_targets = get_predictions_and_targets(
        model, test_loader, device
    )
    test_metrics = calculate_metrics(test_predictions, test_targets)
    test_metrics["loss"] = avg_test_loss

    return test_metrics
