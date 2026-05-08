import torch

def verhulst(dy_dt:torch.Tensor,t:float,y:float,parameters:dict[str,float])->float:
    kappa = parameters["kappa"]
    L = parameters["L"]
    return dy_dt - kappa * y * (1 - y / L)

