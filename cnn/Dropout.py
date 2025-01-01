import torch


class Dropout:
    """
    Dropout Layer.
    Math Formula:
        y = x * mask / (1 - p) (during training)
        y = x (during inference)
    """
    def __init__(self,p=0.5):
        self.p = p # Dropout probability
        self.mask = None #Binary mask
    
    def __call__(self, x, training=True):
        if training:
            #Create a binary mask with probability (1 - p) of keeping an element
            self.mask = (torch.rand_like(x) > self.p).float()
            # Scale the output by (1 - p) to maintain the expected value
            return x * self.mask / (1 - self.p)
        else:
            # During inference, return the input as is
            return x

input = torch.randn(10,5)
dropout = Dropout(p=0.3)

output_train = dropout(input, training=True)
print("Dropout Output (Training):\n", output_train)

output_inference = dropout(input, training=False)
print("Dropout Output (Inference):\n", output_inference)