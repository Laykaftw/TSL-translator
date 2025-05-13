import torch
import torch.nn as nn
import torchvision.models as models

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes, input_size=128, hidden_size=256, dropout_rate=0.5,
                 bidirectional=True, num_lstm_layers=2):
        super(SignLanguageModel, self).__init__()

        # --- DEBUG PRINT ---
        print("  [Model Init] Starting...")
        # --- END DEBUG PRINT ---

        # --- CNN Feature Extractor (Example using ResNet18) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- MODIFY THE FIRST CONV LAYER for 1 input channel (grayscale) ---
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1, # Changed from 3 to 1
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # --- END MODIFICATION ---

        modules = list(resnet.children())[:-1] # Remove final FC layer
        self.cnn_features = nn.Sequential(*modules)

        # --- DEBUG PRINT ---
        print(f"  [Model Init] CNN features defined. Determining output size with input_size={input_size}...")
        # --- END DEBUG PRINT ---

        # Get the output feature size from the CNN dynamically
        cnn_output_features = 0 # Initialize
        try:
            with torch.no_grad(): # No need to track gradients here
                # --- DEBUG PRINT ---
                print(f"    [Model Init] Creating dummy input (1, 1, {input_size}, {input_size})...")
                # --- END DEBUG PRINT ---
                dummy_input = torch.randn(1, 1, input_size, input_size) # Batch=1, Channels=1, H, W

                # --- DEBUG PRINT ---
                print("    [Model Init] Passing dummy input through self.cnn_features...")
                # --- END DEBUG PRINT ---
                dummy_output = self.cnn_features(dummy_input) # <--- PROBLEM LIKELY HERE
                # --- DEBUG PRINT ---
                print("    [Model Init] Dummy input passed through CNN.")
                # --- END DEBUG PRINT ---

                cnn_output_features = dummy_output.view(dummy_output.size(0), -1).shape[1]
                # --- DEBUG PRINT ---
                print(f"    [Model Init] Determined CNN output features: {cnn_output_features}")
                # --- END DEBUG PRINT ---

        except Exception as e:
            print(f"  [Model Init] ERROR determining CNN output size: {e}")
            print("  [Model Init] Check if input_size is compatible with the CNN architecture.")
            # Optionally, raise the error or set a default size if appropriate
            # raise e
            cnn_output_features = 512 # Fallback size (adjust if needed)
            print(f"  [Model Init] Using fallback CNN output features: {cnn_output_features}")


        # --- LSTM Layer ---
        # --- DEBUG PRINT ---
        print(f"  [Model Init] Defining LSTM with input_size={cnn_output_features}...")
        # --- END DEBUG PRINT ---
        self.lstm = nn.LSTM(
            input_size=cnn_output_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )

        # --- Classifier ---
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_size, num_classes)

        # --- DEBUG PRINT ---
        print("  [Model Init] LSTM and Classifier defined. Initialization complete.")
        # --- END DEBUG PRINT ---

    def forward(self, x):
        # x shape: (batch, seq_len, channels=1, height, width)
        batch_size, seq_len, C, H, W = x.size()

        # Reshape for CNN: (batch * seq_len, C, H, W)
        cnn_in = x.view(batch_size * seq_len, C, H, W)

        # Pass through CNN
        cnn_out = self.cnn_features(cnn_in)
        cnn_out = cnn_out.view(batch_size * seq_len, -1) # Flatten features

        # Reshape for LSTM: (batch, seq_len, cnn_output_features)
        lstm_in = cnn_out.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_in)

        # Use output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Classify
        out = self.dropout(last_time_step_out)
        out = self.fc(out)

        return out

# Example usage
if __name__ == '__main__':
    print("Testing SignLanguageModel initialization directly...")
    try:
        model = SignLanguageModel(num_classes=10, input_size=128)
        print("\nModel Structure:")
        # print(model) # Print model structure
        # Test with dummy data
        print("\nTesting forward pass with dummy data...")
        dummy_batch = torch.randn(4, 16, 1, 128, 128) # Batch=4, Seq=16, Channels=1, H=128, W=128
        output = model(dummy_batch)
        print("Forward pass successful!")
        print("Output shape:", output.shape) # Should be (4, 10)
    except Exception as e:
        print(f"\nERROR during direct model test: {e}")
        import traceback
        traceback.print_exc()
