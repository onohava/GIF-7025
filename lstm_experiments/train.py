import os
import json
import argparse
import time
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_loader import EarthQuakeDataLoader
from model import LSTMModel
from constants import *


def train_model(model, train_loader, test_loader, model_name, lookback, target_scaler):
    model_name_with_lb = f"{model_name}_lb{lookback}"
    print(f"\n--- Training {model_name_with_lb} ---")

    save_path = os.path.join(MODEL_SAVE_DIR, model_name_with_lb)
    os.makedirs(save_path, exist_ok=True)

    model_file = os.path.join(save_path, f'{model_name_with_lb}.pth')
    history_file = os.path.join(save_path, f'{model_name_with_lb}_history.json')
    scaler_file = os.path.join(save_path, f'{model_name_with_lb}_scaler.gz')

    # we need this later for inference/plotting to inverse_transform predictions
    print(f"Saving scaler to {scaler_file}...")
    joblib.dump(target_scaler, scaler_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = inputs.float()
            labels = labels.float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                labels = labels.float().view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    end_time = time.time()
    training_time = end_time - start_time

    history['training_time_sec'] = training_time
    history['best_val_loss'] = best_val_loss

    with open(history_file, 'w') as f:
        json.dump(history, f)

    print(f"Training Complete. Best Val Loss: {best_val_loss:.6f}")


def main(lookback_value):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    LOOKBACK = lookback_value

    data = EarthQuakeDataLoader.load_and_prep_data(DATA_FILEPATH, LOOKBACK, test_split=TEST_SPLIT)
    X_train, y_train, X_test, y_test, num_features, target_scaler = data

    if X_train is None:
        print("Error: No training data found.")
        return

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    lstm_model = LSTMModel(
        input_size=num_features,
        **MODEL_DIMS
    )

    train_model(lstm_model, train_loader, test_loader, 'lstm_model', LOOKBACK, target_scaler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM and Bi-LSTM models.")
    parser.add_argument(
        '--lookback',
        type=int,
        default=10,
        help='The lookback period (sequence length) for the models.'
    )
    args = parser.parse_args()

    main(args.lookback)