import torch
import os

def find_model(model_name):
    """
    Finds a pre-trained Latte model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find Latte checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Using Ema!')
        checkpoint = checkpoint["ema"]
    else:
        print('Using model!')
        try:
            checkpoint = checkpoint['model']
        except:
            checkpoint = checkpoint['state_dict']
    
    return checkpoint