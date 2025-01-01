import torch
import torch.nn.functional as F

class Conv1d:
    """
    1D Convolution Layer:
    Math Formula:
        Output[b, c_out, i] = Sum_c_in Sum_k (Input[b, c_in, stride * i + k] * Weight[c_out, c_in, k]) + Bias[c_out]
    
    Using: time-series, speech models.
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(self.kernel_size,tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = torch.randn(out_channels,in_channels,kernel_size)
        self.bias = torch.randn(out_channels)
    
    def __call__(self,x):
        # Add padding to the input (only the last dimension)
        x_padded = F.pad(x,(self.padding, self.padding))
        
        # Get input and output dimensions
        batch_size, in_channel, in_length = x_padded.shape

        #[input_size + 2*padding - kernel_size]//stride + 1
        out_lenght = (in_length - self.kernel_size) // self.stride + 1
        
        #Initialize output tensor
        output = torch.zeros(batch_size,self.out_channels,out_lenght)
        
        #Perform Convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(out_lenght):
                    start = i * self.stride
                    end = start + self.kernel_size
                    #Patch Extraction
                    patch = x_padded[b,:,start:end]
                    output[b,c_out,i] = torch.sum(patch * self.weights[c_out]) + self.bias[c_out] #torch.sum(element-wise multiplication) + bias
        return output
class Conv2d:
    """
    2D Convolution Layer:
    Math Formula:
        Output[b, c_out, i,j] = Sum_c_in Sum_kh Sum_kw (Input[b, c_in, stride * i + kh, stride * j + kw] * Weight[c_out, c_in, kh, kw]) + Bias[c_out]
    Using: Extract feature from image.
    """
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=1):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size[0] if isinstance(self.kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.bias = torch.randn(out_channel)
    
    def __call__(self,x):
        #Add padding to the input (last two dimension)
        x_padded = F.pad(x,(self.padding,self.padding,self.padding,self.padding))

        #Get input and output dimensions
        batch_size, in_channels, in_height,in_width = x_padded.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size, self.out_channel, out_height, out_width)

        for b in range(batch_size):
            for c_out in range(self.out_channel):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b,c_out,i,j] = torch.sum(patch * self.weights[c_out]) + self.bias[c_out]
        return output
    
class Conv3d:
    """
    3D Convolution Layer.
    Math Formula:
        Output[b,c_out,i,j,k] = Sum_c_in Sum_kd Sum_kh Sum_kw (Input[b,c_in,stride * i, kd, stride * j + kh, stride * k + kw] + Weight[c_out, c_in, kd, kh, kw]) + Bias[c_out]
    Using:
    +Video Data(temporal + spatial dimensions)
    +Volumetric Data(3D medical images like CT/MRI scans).
    Scientific Data(3D simulations or climate models)
    3D Object Recognition(point clouds or 3D shapes)
    """
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=1):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size[0] if isinstance(self.kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = torch.randn(out_channel,in_channel,kernel_size,kernel_size,kernel_size)
        self.bias = torch.randn(out_channel)
    
    def __call__(self,x):
        x_padded = F.pad(x,(self.padding,self.padding,self.padding,self.padding,self.padding,self.padding))

        batch_size, in_channel, in_depth,in_height, in_width = x_padded.shape
        out_depth = (in_depth - self.kernel_size) // self.stride + 1
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_witdh = (in_width - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size,self.out_channel,out_depth,out_height,out_witdh)

        for b in range(batch_size):
            for c_out in range(self.out_channel):
                for i in range(out_depth):
                    for j in range(out_height):
                        for k in range(out_witdh):
                            d_start = i * self.stride
                            h_start = j * self.stride
                            w_start = k * self.stride
                            d_end = d_start + self.kernel_size
                            h_end = h_start + self.kernel_size
                            w_end = w_start + self.kernel_size

                            patch = x_padded[b,:,d_start:d_end,h_start:h_end,w_start:w_end]
                            output[b,c_out,i,j,k] = torch.sum(patch * self.weight[c_out]) + self.bias[c_out]
        return output

input = torch.randn(1,16,16)
model = Conv1d(16,8,3)
print(model(input))