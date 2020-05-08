import torch
from torch import optim
from torch.optim.optimizer import Optimizer, required

# SGD
def optm_SGD(cfg, params):
    LR = cfg['LR'] # initial LR
    MOMENTUM = cfg['MOMENTUM']
    optimizer = optim.SGD(params, lr = LR, momentum = MOMENTUM)
    return optimizer

# AdaDelta
def optm_Adadelta(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0
    optimizer = optim.Adadelta(params, lr = LR, rho=0.9, eps=1e-06)

    return optimizer

# AdaGrad
def optm_Adagrad(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10
    optimizer = optim.Adagrad(params, lr = LR, initial_accumulator_value=0, eps=1e-10)

# Adam
def optm_Adam(cfg, params):
    LR = cfg['LR'] # initial LR
    MOMENTUM = cfg['MOMENTUM']
    # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    optimizer = optim.Adam(params, lr = LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

# AmsGrad
def optm_AmsGrad(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True
    optimizer = optim.Adam(params, lr = LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    return optimizer
                          

# AdamW
def optm_AdamW(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    optimizer = optim.AdamW(params, lr = LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    return optimizer

# AmsGradW
def optm_AmsGradW(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True
    optimizer = optim.AdamW(params, lr = LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    return optimizer

# Adamax
def optm_Adamax(cfg, params):
    LR = cfg['LR'] # initial LR
    # params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    optimizer = optim.Adamax(params, lr = LR, betas=(0.9, 0.999), eps=1e-08)
    return optimizer

# RMSprop
def optm_RMSprop(cfg, params):
    LR = cfg['LR'] # initial LR
    MOMENTUM = cfg['MOMENTUM']
    # params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False
    optimizer = optim.RMSprop(params, lr = LR, momentum = MOMENTUM, alpha=0.99, eps=1e-08, centered=False)
    return optimizer

# AdaBound
def optm_AdaBound(cfg, params):
    import adabound
    optimizer = adabound.AdaBound(params, lr=cfg['LR'], final_lr=cfg['TARGET_LR'])
    return optimizer

# RAdm
def optm_RAdam(cfg, params):
    from radam import RAdam
    # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True
    optimizer = RAdam(params, lr=cfg['LR'], betas=(0.9, 0.999), eps=1e-8, degenerated_to_sgd=True)
    return optimizer

# lookAhead
def optm_LookAhead(cfg, params):
    from lookahead import Lookahead
    if not isinstance(params, Optimizer):
        # default using Adam
        params = optm_Adam(cfg, params)
    optimizer = Lookahead(params, k=5, alpha=0.5)
    return optimizer

# Ranger
def optm_Ranger(cfg, params):
    from ranger import Ranger 
    # lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0
    optimizer = Ranger(params, lr=cfg['LR'], alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5)
    return optimizer