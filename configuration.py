class CONFIG:
    # model
    NUM_CLASSES = 3
    BACKBONE = 'efficientnet-b1'
    DESCRIPTION = 'unet-efficientnet_b1-224x224-aug2-split2'
    MODEL_NAME    = 'Unet'
    SAVE_PATH = './checkpoint.pth'
    SAVE_PATH_BEST = './best.pth'
    LOAD_PATH = './checkpoint.pth'

    # training
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 128
    n_accumulate  = max(1, 32//TRAIN_BATCH_SIZE)

    # scheduler
    SCHEDULER = 'CosineAnnealingLR'
    MIN_LR = 1e-6
    T_0 = 25
    T_MAX = int(30000/TRAIN_BATCH_SIZE*EPOCHS)+50

    # optimizer
    WEIGHT_DECAY = 1e-6
    LR = 2e-3

    # misc
    USE_WANDB = True
