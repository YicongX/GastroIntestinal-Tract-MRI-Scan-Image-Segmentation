from imports import *
import configuration

def fetch_scheduler(optimizer):
    scheduler_name = CONFIG.SCHEDULER
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG.T_MAX, 
                                                   eta_min=CONFIG.MIN_LR)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG.T_0, 
                                                             eta_min=CONFIG.MIN_LR)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CONFIG.MIN_LR,)
    elif scheduler_name == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif scheduler_name == None:
        return None
        
    return scheduler