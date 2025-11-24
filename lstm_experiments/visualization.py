import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from constants import *
from sklearn.metrics import mean_squared_error, mean_absolute_error


import torch
from model import LSTMModel, BiLSTMModel

from data_loader import EarthQuakeDataLoader


def create_summary_plots(lookback_values, save_dir):
    results = []

    for lb in lookback_values:
        for model_type in ['lstm', 'bilstm']:
            model_name = f'{model_type}_model_lb{lb}'
            history_file = os.path.join(MODEL_SAVE_DIR, model_name, f'{model_name}_history.json')

            with open(history_file, 'r') as f:
                history = json.load(f)

            results.append({
                'Model Type': 'Bi-LSTM' if model_type == 'bilstm' else 'LSTM',
                'Lookback': lb,
                'Model': f"{'Bi-LSTM' if model_type == 'bilstm' else 'LSTM'} (LB={lb})",
                'Best Validation Loss': history['best_val_loss'],
                'Training Time (sec)': history['training_time_sec']
            })

    df_results = pd.DataFrame(results)

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



def plot_training_history(history_lstm_path, history_bilstm_path, save_path, lookback):
    with open(history_lstm_path, 'r') as f:
        history_lstm = json.load(f)
    with open(history_bilstm_path, 'r') as f:
        history_bilstm = json.load(f)

    plt.figure(figsize=(14, 7))

    plt.plot(history_lstm['loss'], label='LSTM Train Loss', color='blue', linestyle='--')
    plt.plot(history_lstm['val_loss'], label='LSTM Val Loss', color='blue', linewidth=2)
    plt.plot(history_bilstm['loss'], label='Bi-LSTM Train Loss', color='orange', linestyle='--')
    plt.plot(history_bilstm['val_loss'], label='Bi-LSTM Val Loss', color='orange', linewidth=2)

    plt.title(f'Model Loss Comparison (Lookback={lookback})', fontsize=16)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(save_path)
    print(f"Saved training history plot to {save_path}")
    plt.close()


def plot_predictions(model, model_path, X_test, y_test, model_name, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    predictions = []
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        outputs = model(X_test_tensor)

        predictions = outputs.cpu().numpy()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Magnitude', color='black', alpha=0.7)
    plt.plot(predictions, label='Predicted Magnitude', color='red', linestyle='--', alpha=0.8)

    plt.title(f'{model_name} - Predictions vs. Actuals (Scaled Data)', fontsize=16)
    plt.ylabel('Scaled Magnitude', fontsize=12)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.grid(True)

    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    X_train, y_train, X_test, y_test, num_features = data

    if X_test is None:
        print("Exiting visualization due to data loading error.")
        return

    model_name_lstm = f'lstm_model_lb{LOOKBACK}'
    model_name_bilstm = f'bilstm_model_lb{LOOKBACK}'

    history_lstm_path = os.path.join(MODEL_SAVE_DIR, model_name_lstm, f'{model_name_lstm}_history.json')
    history_bilstm_path = os.path.join(MODEL_SAVE_DIR, model_name_bilstm, f'{model_name_bilstm}_history.json')
    history_save_path = os.path.join(VISUALS_DIR, f'training_loss_comparison_lb{LOOKBACK}.png')

    plot_training_history(history_lstm_path, history_bilstm_path, history_save_path, LOOKBACK)

    lstm_model_path = os.path.join(MODEL_SAVE_DIR, model_name_lstm, f'{model_name_lstm}.pth')  # .pth file
    lstm_pred_save_path = os.path.join(VISUALS_DIR, f'lstm_predictions_lb{LOOKBACK}.png')
    lstm_model = LSTMModel(input_size=num_features, **MODEL_DIMS)

    plot_predictions(lstm_model, lstm_model_path, X_test, y_test, f'LSTM (Lookback={LOOKBACK})', lstm_pred_save_path)

    bilstm_model_path = os.path.join(MODEL_SAVE_DIR, model_name_bilstm, f'{model_name_bilstm}.pth')  # .pth file
    bilstm_pred_save_path = os.path.join(VISUALS_DIR, f'bilstm_predictions_lb{LOOKBACK}.png')
    bilstm_model = BiLSTMModel(input_size=num_features, **MODEL_DIMS)

    plot_predictions(bilstm_model, bilstm_model_path, X_test, y_test, f'Bi-LSTM (Lookback={LOOKBACK})',
                     bilstm_pred_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LSTM model performance.")
    parser.add_argument(
        '--lookback',
        type=int,
        default=10,
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