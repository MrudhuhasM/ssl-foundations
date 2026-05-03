import timm
import torch

def get_encoder(checkpoint_path=None):
    if checkpoint_path is None:
        model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, num_classes=0)
    else:
        model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, num_classes=0)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model