import os

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

set_deterministic(42)

# Step 0: Data analysis
recomendded_clusters = full_analysis()

# # Main pipeline execution
# Improved Well Log VS Prediction Pipeline
# ## Configuration - Improved
base_config = {
    "sequence_length": 15,
    "mask_value": -1.0,
    "num_epochs": 500,
    "patience": 150,
    "target_feature": "VS",
    "clusters": recomendded_clusters,
}
# ## Features Combinations - MORE FEATURES
print("=" * 80)
print("IMPROVED WELL LOG VS PREDICTION PIPELINE")
print("=" * 80)
# Feature combinations - MORE FEATURES
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

# ## Step 1: Load Data
print("\n" + "=" * 80)
print("STEP 1: LOADING WELL DATA")
print("=" * 80)

wells = read_all_wells_with_dept_to_list(features="all")
well_dfs = filter_commom_features(wells, ignore=["VS"])
# ### Add derived features
print("Adding derived features...")
well_dfs = [add_derived_features(df) for df in well_dfs]

wells_with_vs = [df for df in well_dfs if "VS" in df.columns]
wells_without_vs = [df for df in well_dfs if "VS" not in df.columns]

print(f"Total wells loaded: {len(well_dfs)}")
print(f"Wells with VS: {len(wells_with_vs)}")
print(f"Wells without VS: {len(wells_without_vs)}")

# ## Step 2: Run Cross-Fold Validation
print("\n" + "=" * 80)
print("STEP 2: CROSS-FOLD VALIDATION")
print("=" * 80)

existing_experiments = [
    d for d in os.listdir("experiments") if d.startswith("experiment_")
]

if existing_experiments:
    print(f"\nFound {len(existing_experiments)} existing experiment(s):")
    for i, exp_dir in enumerate(existing_experiments):
        print(f" {i + 1}. {exp_dir}")

    response = (
        input("\nDo you want to use existing experiments(s) (y/n): ").strip().lower()
    )

    if response == "y":
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

        print(f"\nUsing existing experiment: {experiment_dir}")
        best_config = load_best_configuration(experiment_dir)
    else:
        print("\nRunning new cross-fold validation...")
        experiment = CrossFoldHyperparameterExperiment(base_config)
        results = experiment.run_cross_fold_experiments(
            feature_combinations, wells_with_vs
        )
        experiment_dir = experiment.results_dir
        best_config = load_best_configuration(experiment_dir)
else:
    print("\nNo existing experiments found. Running cross-fold validation...")
    experiment = CrossFoldHyperparameterExperiment(base_config)
    results = experiment.run_cross_fold_experiments(feature_combinations, wells_with_vs)
    experiment_dir = experiment.results_dir
    best_config = load_best_configuration(experiment_dir)
# ## STEP 3: Train Final Model
print("\n" + "=" * 80)
print("STEP 3: TRAINING FINAL MODEL")
print("=" * 80)

final_output_dir = os.path.join(experiment_dir, "final_model")
trainer = FinalModelTrainer(best_config, base_config, output_dir=final_output_dir)
trained_clusters = trainer.train_final_model(
    wells_with_vs, target_feature=base_config["target_feature"]
)
# ## STEP 4: Make Predictions
print(f"\n" + "=" * 80)
print("STEP 4: MAKING PREDICTIONS")
print("=" * 80)

all_results = trainer.predict_on_wells(
    trained_clusters,
    wells_without_vs,
    wells_with_vs,
    target_feature=base_config["target_feature"],
)
# ## STEP 5: Generate Plots and Reports
print("\n" + "=" * 80)
print("STEP 5: GENERATING REPORTS AND PLOTS")
print("=" * 80)

trainer.plot_predictions(all_results)
trainer.generate_summary_report(all_results, best_config)

# # Done
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {trainer.output_dir}/")
print(f"  - Final model: final_model.pth")
print(f"  - Scaler: final_scaler.pkl")
print(f"  - Predictions: all_predictions.json")
print(f"  - Summary: SUMMARY_REPORT.txt")
print(f"  - Plots: plots/")
print("\n" + "=" * 80)
