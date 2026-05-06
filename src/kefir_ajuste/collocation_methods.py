import numpy as np
import torch

def all_data_collocation(X:torch.Tensor,y:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    """
    Use the full dataset as collocation points.

    This function returns the input data without modification, meaning all
    available samples are used as collocation points.

    Parameters
    ----------
    X : torch.Tensor
        Input feature tensor.
    y : torch.Tensor
        Target tensor.
    args : dict of str to int
        Additional configuration arguments (not used).

    Returns
    -------
    X : torch.Tensor
        Unmodified input features.
    y : torch.Tensor
        Unmodified target values.
    """
    return X,y

def identity_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    """
    Return training data without modification.

    This function is equivalent to an identity mapping and is used when
    no collocation strategy is applied.

    Parameters
    ----------
    X_train : torch.Tensor
        Training input features.
    y_train : torch.Tensor
        Training target values.
    args : dict of str to int
        Additional configuration arguments (not used).

    Returns
    -------
    X_train : torch.Tensor
        Unmodified training inputs.
    y_train : torch.Tensor
        Unmodified training targets.
    """
    return X_train, y_train

def random_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    """
    Select a random subset of collocation points.

    This function randomly samples a subset of the training data to be used
    as collocation points.

    Parameters
    ----------
    X_train : torch.Tensor
        Training input features.
    y_train : torch.Tensor
        Training target values.
    args : dict of str to int
        Configuration dictionary containing:
        - ``collocation_size`` : int
            Number of samples to select.
        - ``seed`` : int or None
            Random seed for reproducibility.

    Returns
    -------
    X_sub : torch.Tensor
        Randomly selected input features.
    y_sub : torch.Tensor
        Randomly selected target values.

    Notes
    -----
    Sampling is done without replacement.
    """
    collocation_size= args["collocation_size"]
    seed = args["seed"]

    if seed is not None:
        np.random.seed(seed)
    idx = np.random.choice(len(X_train), size=collocation_size, replace=False)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub

def equal_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    """
    Select evenly spaced collocation points from the dataset.

    This function subsamples the dataset using a fixed stride, producing a
    uniformly spaced subset of the training data.

    Parameters
    ----------
    X_train : torch.Tensor
        Training input features.
    y_train : torch.Tensor
        Training target values.
    args : dict of str to int
        Configuration dictionary containing:
        - ``collocation_skip`` : int
            Step size for subsampling.

    Returns
    -------
    X_sub : torch.Tensor
        Subsampled input features.
    y_sub : torch.Tensor
        Subsampled target values.

    Notes
    -----
    The first sample (index 0) is excluded due to starting index = 1.
    """
    collocation_skip=args["collocation_skip"]
    idx = np.arange(1, len(X_train), collocation_skip)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub