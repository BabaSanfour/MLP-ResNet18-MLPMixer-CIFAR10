import torch
import numpy as np
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor( out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        return nn.functional.linear(input, self.weight, self.bias)


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        hidden_layers = nn.ModuleList(modules=None)
        input_layer = Linear(input_size, hidden_sizes[0])
        hidden_layers.append(input_layer)
        for i in range(len(hidden_sizes)-1):
            hidden_layer = Linear(hidden_sizes[i], hidden_sizes[i+1])
            hidden_layers.append(hidden_layer)
        output_layer = Linear(hidden_sizes[-1], num_classes) 
        return (hidden_layers, output_layer)
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        if  activation == "tanh":
            return nn.Tanh()(inputs)
        elif activation == "relu":
            return nn.ReLU()(inputs)
        else:
            return nn.Sigmoid()(inputs)

    def _initialize_linear_layer(self, module: Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        norm_value =1/np.sqrt(module.weight.shape[0])
        nn.init.uniform_(module.weight, a=-norm_value, b=norm_value)
        nn.init.constant_(module.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        X = torch.flatten(images, start_dim=1)
        for layer in self.hidden_layers:
            X = layer.forward(X)
            X = self.activation_fn(self.activation, X)
        X = self.output_layer.forward(X)

        return X
