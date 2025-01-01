import torch
import torch.nn as nn

"""
LSTM Overview:
    LSTM is designed to solve gradient problems in RNNs's gates (input, forget, output) and a cell state.
    It can learn long-length sequence better than RNNs and can outperforms than GRUs in complex tasks.

LSTM's gates:
    Input gates: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
    Forget gates: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
    Candidate Cell Gate: Can_t = tanh(W_C * [h_{t-1}, x_t] + b_C)

    Cell State: C_t = f_t * C_{t-1} + i_t * Can_t
    Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
    Hidden Gate: h_t = o_t * tanh(C_t)
"""


class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        #Initial the weights and biases of Gates (Input, Forget, Output, Candidate)
        self.W_f = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size)) #Forget gates weights
        self.W_i = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size)) #Input gates weights
        self.W_o = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size)) # Output gates Weights
        self.W_C = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size)) # Candidate gates Weights

        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        #Output Layer
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size)) #its weight
        self.b_y = nn.Parameter(torch.zeros(output_size)) #its bias

    def __call__(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size) #hidden state
        C = torch.zeros(batch_size, self.hidden_size) #cell state
        
        output = []

        for t in range(seq_len):
            combined = torch.cat((h, x[:,t,:]), dim=1)

            forget_gate = torch.sigmoid(combined @ self.W_f + self.b_f)
            input_gate = torch.sigmoid(combined @ self.W_i + self.b_i)
            
            #Candidate cell state
            C_tilde = torch.tanh(combined @ self.W_C + self.b_c)

            #Update cell state
            C = forget_gate * C + input_gate * C_tilde

            output_gate = torch.sigmoid(combined @ self.W_o + self.b_o)

            h = output_gate * torch.tanh(C)

            y = h @ self.W_hy + self.b_y
            output.append(y)
        
        return torch.stack(output, dim=1)

#Basic LSTM Neural Network

class LSTM_Network(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

try:
    input = torch.randn(16,10,5)
    lstm = LSTM_NN(input_size=5,hidden_size=10,output_size=1)
    output1 = lstm(input)
    print(f"The LSTM of PyTorch's output: {output1.shape} ".format())

    lstm_nn = LSTM(input_size=5, hidden_size=10, output_size=1)
    output2 = lstm_nn(input)
    print(f"The LSTM of me's output: {output2.shape}")

except Exception as e:
    print(f"{e}")