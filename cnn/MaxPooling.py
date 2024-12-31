import torch

class MaxPooling1D:
    """
    MaxPooling1D Layer:
    Math Formula:
        Output[b,c_out,i] = Max_k (Input[b,c_out,stride * i + k])
    """
    def __init__(self,kernel_size,stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x):
        batch_size,channels,in_lenght = x.shape
        out_lenght = (in_lenght - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size, channels, out_lenght)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_lenght):
                    start = i * self.stride
                    end = start + self.kernel_size
                    patch = x[b, c, start:end]
                    output[b, c, i] = torch.max(patch)
        
        return output

class MaxPooling2D:
    """
    MaxPooling2D Layer:
    Math Formula:
        Output[b, c_out, i,j] = Max_kh Max_kw (Input[b, c_out, stride * i + kh, stride * j + kw])
    """
    def __init__(self,kernel_size,stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size,channels,out_height,out_width)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        patch = x[b, c,h_start:h_end ,w_start:w_end]
                        output[b, c, i, j] = torch.max(patch)
        return output

class MaxPooling3D:
    """
    MaxPooling3D Layer:
    Math Formula:
        Output[b, c_out,i ,j ,k] = Max_kd Max_kh Max_kw (Input[b, c_out, stride * i + kd,stride * j + kh, stride * k + kw])
    """
    def __init__(self,kernel_size,stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self,x):
        batch_size, channels, in_depth, in_height, in_width = x.shape
        out_depth = (in_depth - self.kernel_size) // self.stride + 1
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size ,channels ,out_depth ,out_height ,out_width)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_depth):
                    for j in range(out_height):
                        for k in range(out_width):
                            d_start = i * self.stride
                            h_start = j * self.stride
                            w_start = k * self.stride
                            d_end = d_start + self.kernel_size
                            h_end = h_start + self.kernel_size
                            w_end = w_start + self.kernel_size

                            patch = x[b, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            output[b, c, i, j, k] = torch.max(patch)
        return output