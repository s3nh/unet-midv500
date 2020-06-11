import os 
import torch.nn as nn 
import torch.nn.init as init 
from src.unet import UNet



def main():
    os.makedirs('exported', exist_ok = True)
    torch_model = '../trained_models/unet_best_9.pt'
    x = torch.randn(4, 1, 768, 768, requires_grad = True) 
    torch_out = torch_model(x)
    torch.onnx.export(torch_model, 
            x, 
            'unet_9.onnx', 
            export_params=True, 
            opset_version = 10, 
            do_constant_folding = True, 
            input_names = ['input'], 
            output_names = ['output'], 
            dynamic_axes = {'input' : {0: 'batch_size'}, 
                            'output' : {0 : 'batch_size'}})
            
if __name__ == "__main__":
    main()
