import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Enhanced BiLSTM model with Attention Mechanism and Dropout
class EnhancedBiLSTMModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMModelWithAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_layer_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # *2 for bidirectional LSTM and 2 timeframes

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        lstm_out = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        predictions = self.linear(lstm_out)
        return predictions

# Function to fetch historical data from Binance
def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        return df
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

# Feature Engineering: Adding Technical Indicators
def add_technical_indicators(df):
    df['SMA_5'] = df['price'].rolling(window=5).mean()
    df['SMA_15'] = df['price'].rolling(window=15).mean()
    df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['Bollinger_Upper'] = df['SMA_15'] + (df['price'].rolling(window=15).std() * 2)
    df['Bollinger_Lower'] = df['SMA_15'] - (df['price'].rolling(window=15).std() * 2)
    df = df.dropna()  # Drop rows with NaN values after adding indicators
    return df

# Prepare the dataset with sliding window approach
def prepare_dataset(symbols, sequence_length=10):
    all_data = []
    for symbol in symbols:
        df = get_binance_data(symbol)
        df = add_technical_indicators(df)
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df[['price', 'SMA_5', 'SMA_15', 'EMA_10', 'Bollinger_Upper', 'Bollinger_Lower']])
        
        for i in range(sequence_length, len(scaled_data) - 20):  # Consider the 20-minute prediction
            seq = scaled_data[i-sequence_length:i]
            label_10 = scaled_data[i+10, 0]  # Only the price column
            label_20 = scaled_data[i+20, 0]
            label = torch.FloatTensor([label_10, label_20])
            all_data.append((seq, label))
    return all_data, scaler

# Define the training process with early stopping
def train_model(model, data, epochs=100, lr=0.001, sequence_length=10, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for seq, label in data:
            seq = torch.FloatTensor(seq).view(1, sequence_length, -1).to(model.linear.weight.device)
            label = label.view(1, -1).to(model.linear.weight.device)  # Ensure label has the shape [batch_size, 2]

            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_enhanced_bilstm_model_with_attention.pth")
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedBiLSTMModelWithAttention(input_size=6, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3).to(device)

    # Symbols to train on
    symbols = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARBUSDT']

    # Prepare data
    data, scaler = prepare_dataset(symbols)

    # Train the model
    train_model(model, data)
