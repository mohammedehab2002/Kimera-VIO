import torch
import pytorch_metric_learning.utils.common_functions as common_functions
from vpr_model import VPRModel

def load_model(ckpt_path):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    model.load_state_dict(torch.load(ckpt_path,map_location=torch.device('cpu'))['state_dict'])
    model = model.eval()
    # model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

model=load_model("./cliquemining.ckpt")
img_height=238
img_width=308
input=torch.randn((1,3,img_width,img_height))
traced_model=model.to_torchscript(method="trace",example_inputs=input)
torch.jit.save(traced_model,f"SALAD_scripted_{img_height}_{img_width}.pt")
