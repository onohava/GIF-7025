import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from constants import *
from model import LSTMModel
from data_loader import EarthQuakeDataLoader


def create_summary_plots(lookback_values, save_dir):
    results = []

    for lb in lookback_values:
        model_name = f'lstm_model_lb{lb}'
        history_file = os.path.join(MODEL_SAVE_DIR, model_name, f'{model_name}_history.json')

        if not os.path.exists(history_file):
            print(f"Warning: History file not found for {model_name}")
            continue

        with open(history_file, 'r') as f:
            history = json.load(f)

        results.append({
            'Model Type': 'LSTM',
            'Lookback': lb,
            'Model': f"LSTM (LB={lb})",
            'Best Validation Loss': history['best_val_loss'],
            'Training Time (sec)': history['training_time_sec']
        })

    if not results:
        print("No results found to plot.")
        return

    df_results = pd.DataFrame(results)

    # Plot 1: Validation Loss
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_results,
        x='Lookback',
        y='Best Validation Loss',
        hue='Model Type',
        palette='viridis'
    )
    plt.title('Experiment Summary: Model Performance (Lower is Better)', fontsize=16)
    plt.ylabel('Best Validation Loss (MSE)', fontsize=12)
    plt.xlabel('Lookback (Sequence Length)', fontsize=12)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path_loss = os.path.join(save_dir, 'summary_performance_comparison.png')
    plt.savefig(save_path_loss)
    print(f"Saved performance summary plot to {save_path_loss}")
    plt.close()

    # Plot 2: Training Time
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_results,
        x='Lookback',
        y='Training Time (sec)',
        hue='Model Type',
        palette='plasma'
    )
    plt.title('Experiment Summary: Model Training Speed', fontsize=16)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xlabel('Lookback (Sequence Length)', fontsize=12)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path_time = os.path.join(save_dir, 'summary_speed_comparison.png')
    plt.savefig(save_path_time)
    print(f"Saved speed summary plot to {save_path_time}")
    plt.close()


def plot_training_history(history_lstm_path, save_path, lookback):
    has_lstm = os.path.exists(history_lstm_path)

    plt.figure(figsize=(14, 7))

    if has_lstm:
        with open(history_lstm_path, 'r') as f:
            history_lstm = json.load(f)
        plt.plot(history_lstm['loss'], label='LSTM Train Loss', color='blue', linestyle='--')
        plt.plot(history_lstm['val_loss'], label='LSTM Val Loss', color='blue', linewidth=2)

    plt.title(f'Model Loss Comparison (Lookback={lookback})', fontsize=16)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(save_path)
    print(f"Saved training history plot to {save_path}")
    plt.close()


def plot_predictions(model, model_path, X_test, y_test, model_name, save_path, target_scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
    else:
        X_test_tensor = X_test.float().to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = outputs.cpu().numpy()

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    predictions_real = target_scaler.inverse_transform(predictions)
    y_test_real = target_scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, predictions_real)
    mae = mean_absolute_error(y_test_real, predictions_real)
    plt.figure(figsize=(14, 7))

    x_axis = range(len(y_test_real))

    plt.scatter(x_axis, y_test_real, label='Actual Magnitude',
                color='black', alpha=0.5, s=15)

    plt.scatter(x_axis, predictions_real, label='Predicted Magnitude',
                color='red', alpha=0.5, s=15, marker='x')

    plt.ylabel('Earthquake Magnitude', fontsize=12)
    plt.xlabel('Seismic Events (Index)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add text box with metrics
    plt.text(0.02, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.savefig(save_path)
    print(f"Saved prediction plot to {save_path}")
    plt.close()


def main(lookback_value, run_summary=False, all_lookbacks=None):
    os.makedirs(VISUALS_DIR, exist_ok=True)

    if run_summary:
        if all_lookbacks:
            create_summary_plots(all_lookbacks, VISUALS_DIR)
        else:
            print("Error: `run_summary` is True but `all_lookbacks` was not provided.")
        return

    LOOKBACK = lookback_value

    print(f"\nVisualizing models with LOOKBACK={LOOKBACK}...")

    data = EarthQuakeDataLoader.load_and_prep_data(DATA_FILEPATH, LOOKBACK, test_split=0.2)
    X_train, y_train, X_test, y_test, num_features, target_scaler = data

    if X_test is None:
        print("Exiting visualization due to data loading error.")
        return

    model_name_lstm = f'lstm_model_lb{LOOKBACK}'

    history_lstm_path = os.path.join(MODEL_SAVE_DIR, model_name_lstm, f'{model_name_lstm}_history.json')
    history_save_path = os.path.join(VISUALS_DIR, f'training_loss_comparison_lb{LOOKBACK}.png')

    plot_training_history(history_lstm_path, history_save_path, LOOKBACK)

    lstm_model_path = os.path.join(MODEL_SAVE_DIR, model_name_lstm, f'{model_name_lstm}.pth')
    lstm_pred_save_path = os.path.join(VISUALS_DIR, f'lstm_predictions_lb{LOOKBACK}.png')

    lstm_model = LSTMModel(input_size=num_features, **MODEL_DIMS)

    plot_predictions(
        lstm_model,
        lstm_model_path,
        X_test,
        y_test,
        f'LSTM (Lookback={LOOKBACK})',
        lstm_pred_save_path,
        target_scaler
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LSTM model performance.")
    parser.add_argument(
        '--lookback',
        type=int,
        default=1,
        help='The lookback period (sequence length) of the models to visualize.'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Run the final summary plot for all experiments.'
    )
    parser.add_argument(
        '--all_lookbacks',
        type=str,
        default='10,25,50',
        help='Comma-separated list of lookbacks to use for the summary, e.g., "10,25,50"'
    )

    args = parser.parse_args()

    if args.summary:
        all_lookbacks = [int(lb) for lb in args.all_lookbacks.split(',')]
        main(lookback_value=0, run_summary=True, all_lookbacks=all_lookbacks)
    else:
        main(args.lookback, run_summary=False)