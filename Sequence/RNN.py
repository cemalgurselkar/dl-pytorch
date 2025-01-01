import torch
import torch.nn as nn

"""
RNNs process sequential data by maintaining a hidden state that captures
information from previous time steps.

Math Formula: 
    Hidden State:
        h_t = tanh(Weight_hh * h_{t-1} + W_xh * x_t + b_h)
    
    Output:
        y_t = W_hy * h_t + b_y      
"""

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        #Setting Weight matrices and biases
        # W_xh (Input2hidden), W_hh(hidden2hidden), W_hx(hidden2output)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))

        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size))

        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size) #Initial hidden state
        output = []

        for t in range(seq_len):
            h = torch.tanh(x[:,t,:] @ self.W_xh + h @ self.W_hh + self.b_h)

            y = h @ self.W_hy + self.b_y #Compute output
            output.append(y)
        
        return torch.stack(output, dim=1)

class RNN_Block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self,x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:,-1,:])
        return out

try:
    input = torch.randn(16,10,5)
    rnn1 = RNN_Block(input_size=5,hidden_size=10,output_size=1)
    output1 = rnn1(input)
    rnn2 = RNN(input_size=5, hidden_size=10, output_size=1)
    output2 = rnn2(input)

    print(f"The RNN of PyTorch: {output1.shape}")
    print(f"The RNN of me: {output2.shape}")

except Exception as e:
    print(f"Error message: {e}")