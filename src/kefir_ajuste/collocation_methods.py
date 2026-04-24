import numpy as np

def all_data_collocation(X:np.ndarray,y:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    return X,y

def identity_collocation(X_train:np.ndarray,y_train:np.ndarray,**kwargs)->tuple[np.ndarray,np.ndarray]:
    return X_train, y_train

def random_collocation(X_train:np.ndarray,y_train:np.ndarray,args:dict[str,int|None])->tuple[np.ndarray,np.ndarray]:
    collocation_size= args["collocation_size"]
    seed = args["seed"]

    if seed is not None:
        np.random.seed(seed)
    idx = np.random.choice(len(X_train), size=collocation_size, replace=False)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub

def equal_collocation(X_train:np.ndarray,y_train:np.ndarray,args:dict[str,int])->tuple[np.ndarray,np.ndarray]:
    collocation_skip=args["collocation_skip"]
    idx = np.arange(1, len(X_train), collocation_skip)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    return X_sub,y_sub