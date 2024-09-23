import torch
import torch.nn as nn

class CustomOCRModel(nn.Module):
    def __init__(self,input_shape:int, output_shape:int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
                                            nn.Conv2d(in_channels=input_shape,
                                                      out_channels=32,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=0
                                                      ),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=2)
                                         )
        
        self.conv_block_2 = nn.Sequential(
                                    nn.Conv2d(in_channels=32,
                                                out_channels=64,
                                                kernel_size=3,
                                                padding= 1
                                                ),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        self.conv_block_3 = nn.Sequential(
                                    nn.Conv2d(in_channels=64,
                                                out_channels=128,
                                                kernel_size=3,
                                                padding=0
                                                ),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        
        self.classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(128 * 2 * 2, 64),
                                nn.ReLU(),
                                nn.Linear(64, 128),
                                nn.ReLU(),
                                nn.Linear(128, output_shape),
                                nn.Softmax(dim=1)
                                )   
    
    def forward(self, x): # x: (B,1,28,28)
        x = self.conv_block_1(x) # x: (B, 32, 13, 13)
        x = self.conv_block_2(x) # x: (B, 64, 6, 6)
        x = self.conv_block_3(x) # x: (B, 128, 2, 2)
        x = self.classifier(x) # x: (B, 36)
        return x