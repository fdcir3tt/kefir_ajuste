import numpy as np
import torch

def all_data_collocation(X:torch.Tensor,y:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    return X,y

def identity_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    return X_train, y_train

def random_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    collocation_size= args["collocation_size"]
    seed = args["seed"]

    if seed is not None:
        np.random.seed(seed)
    idx = np.random.choice(len(X_train), size=collocation_size, replace=False)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub

def equal_collocation(X_train:torch.Tensor,y_train:torch.Tensor,args:dict[str,int])->tuple[torch.Tensor,torch.Tensor]:
    collocation_skip=args["collocation_skip"]
    idx = np.arange(1, len(X_train), collocation_skip)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub