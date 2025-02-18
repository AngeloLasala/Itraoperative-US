import torch

def get_numer_parameter(model):
    """
    Get the number of parameters of the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the number of parameters

    Returns
    -------
    int
        The number of parameters of the model, in B
    """
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f"Total parameters: {num_params:.3f} B")
    print(f"Trainable parameters: {trainable_params:.3f} B")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)