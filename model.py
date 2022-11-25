from imports import *

def build_model(model_type = 'smp'):
    model = None
    if model_type == 'smp':
        model = smp.Unet(
        encoder_name=CONFIG.BACKBONE,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CONFIG.NUM_CLASSES,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    
    assert model != None, "Model type invalid"

    model.to(DEVICE)
    return model

def load_model(path, model_type = 'smp'):
    model = build_model(model_type)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model