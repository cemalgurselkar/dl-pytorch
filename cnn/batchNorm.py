import torch
"""
Purpose: BatchNorm normalizes the input to have zero mean and unit variance during training,
which stabilizes and accelerates the training process. During inference, it uses running averages of mean and variance.

Math Formula:
    y = (x - mean) / sqrt(var + epx) * gamma + beta
    
    where x is input.
    where mean is mean of the input (computed per channel).
    where var is variance of the input (computed per channel).
    where gamma is Scale parameter(learnable) and beta is Shift parameter(learnable)
"""

class BatchNorm1D:
    """
    Batch Normalization for 1D inputs (e.g., time-series, sequences).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def __call__(self, x, training=True):
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            #Update runnign averages
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        return x_norm

class BatchNorm2D:
    """
    Batch Normalization for 2D inputs (e.g., images)
    """
    def __init__(self, num_features, eps=1e-5,momentum=0.1):
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def __call__(self, x, training=True):
        if training:
            mean = x.mean(dim=(0,2,3)) #Compute mean over batch, height and width
            var = x.var(dim=(0,2,3), unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, : ,None ,None] + self.eps)
        x_norm = x_norm * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        return x_norm

class BathcNorm3D:
    """
    Batch Normalization for 3D inputs (e.g., volumes, videos)
    """
    def __init__(self,num_features, eps=1e-5, momentum=0.1):
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def __call__(self, x, training=True):
        if training:
            mean = x.mean(dim=(0,2,3,4))
            var = x.var(dim=(0,2,3,4),unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean[None, :, None,None,None]) / torch.sqrt(var[None,:,None,None,None] + self.eps)
        x_norm = x_norm * self.gamma[None,:,None,None,None] + self.beta[None,:,None,None,None]
        return x_norm


input_1d = torch.randn(10, 5)
input_2d = torch.randn(10, 3, 32, 32)
input_3d = torch.randn(10, 3, 16, 16, 16)

bn1d = BatchNorm1D(num_features=5)
bn2d = BatchNorm2D(num_features=3)
bn3d = BathcNorm3D(num_features=3)

output_1d_train = bn1d(input_1d, training=True)
output_2d_train = bn2d(input_2d, training=True)
output_3d_train = bn3d(input_3d, training=True)

output_1d_inference = bn1d(input_1d, training=False)
output_2d_inference = bn2d(input_2d, training=False)
output_3d_inference = bn3d(input_3d, training=False)

print("BatchNorm1D Output (Training):", output_1d_train.shape)
print("BatchNorm2D Output (Training):", output_2d_train.shape)
print("BatchNorm3D Output (Training):", output_3d_train.shape)

print("BatchNorm1D Output (Inference):", output_1d_inference.shape)
print("BatchNorm2D Output (Inference):", output_2d_inference.shape)
print("BatchNorm3D Output (Inference):", output_3d_inference.shape)