import torch


def nan_hook(module, input_, output):
    """Hook to detect NaN values in model outputs.
    
    Args:
        module: The module being monitored
        input_: Input to the module 
        output: Output from the module
    
    Raises:
        RuntimeError: If NaN values are detected in the output
    """
    if ( isinstance(output, torch.Tensor) and (torch.isnan(output).any()) ) or ( isinstance(input_, torch.Tensor) and (torch.isnan(input_).any()) ):
        print(f"NaN detected in {module.__class__.__name__}'s output")
        print(f"Input was: {input_}")
        raise RuntimeError(f"NaN detected in {module.__class__.__name__}")