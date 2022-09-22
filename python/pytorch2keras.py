import torch
from torch.autograd import Variable

from linformer import Linformer
from vit_pytorch.efficient import ViT

import onnx
from onnx_tf.backend import prepare

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - intializing model - - - - - -

efficient_transformer = Linformer(
    dim=1024,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

pt_model = ViT(
    dim=1024,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# - - - - - - load pretrained model - - - - - -
PATH = 'savedModels/ViT_covid_classifier.pt'
pretrained_net = torch.load(PATH, map_location=torch.device('cpu'))
pt_model.load_state_dict(pretrained_net)

dummy_input = Variable(torch.randn(1, 3, 224, 224)) 
torch.onnx.export(pt_model, dummy_input, "savedModels/ViT_covid_classifier.onnx", opset_version=12)

# Load the ONNX file
onnx_model = onnx.load('savedModels/ViT_covid_classifier.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(onnx_model)

tf_rep.export_graph('savedModels/ViT_covid_classifier')
