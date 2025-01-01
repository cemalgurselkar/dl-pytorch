import torch
import torch.nn as nn

"""
GRU is a simpler variant of LSTM with fewer parameters.
It uses update and reset gates to control information flow.

Math Formula:
    Update Gate:
        z_t = sigmoid(Weight_z * [h_{t-1}, x_t] + b_z)
    
    Reset Gate:
        r_t = sigmoid(Weight_r * [h_{t-1}, x_t] + b_r)
    
    Candidate Hidden State:
        CanH_t = tanh(Weight_h * [h_{t-1}, x_t] + b_h)
    
    Hidden State:
        h_t = (1 - z_t) * h_{t-1} + z_t * CanH_t
"""

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_z = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        self.W_r = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        self.W_h = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        #Output Layer
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        outputs = []

        for t in range(seq_len):

            combined = torch.cat((h, x[:,t,:]), dim=1)

            updated_z = torch.sigmoid(combined @ self.W_z + self.b_z)

            updated_r = torch.sigmoid(combined @ self.W_r + self.b_r)

            combined_reset = torch.cat((updated_r * h, x[:,t,:]), dim=1)
            h_title = torch.tanh(combined_reset @ self.W_h + self.b_h)

            updated_h = (1 - updated_z) * h + updated_z * h_title

            y = updated_h @ self.W_hy + self.b_y
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class GRU_Block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:,-1,:])
        return out



try:
    input = torch.randn(16, 10, 5)

    gru = GRU(input_size=5, hidden_size=10, output_size=1)
    output1 = gru(input)
    print(f"The GRU of me{output1.shape}")
    gru2 = GRU_Block(input_size=5,hidden_size=10,output_size=1)
    output2 = gru2(input)
    print(f"The GRU of PyTorch{output2.shape}")

except Exception as e:
    print(f"Error message: {e}")