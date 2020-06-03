import torch.nn as nn 
import torch.nn.init as init 
from src.unet import UNet



def main():
    
    torch.onnx.export(model, 
            x, 
            'unet_midv.onnx', 
            export_params=True, 
            opset_version = 10, 
            do_constant_folding = True, 
            input_names = ['input'], 
            output_names = ['output'], 
            dynamic_axes = {'input' : {0: 'batch_size'}, 
                            'output' : {0 : 'batch_size'}})
            
if __name__ == "__main__":
    main()
