import torch.nn as nn
import torch
from yaml import load_all
from torchvision.models.resnet import resnet50
from transformers import AutoModel
from vit import VisionTransformer
from xbert import BertConfig as AlbefBertConfig, BertModel as AlbefBertModel
from functools import partial
import os
from urllib.request import urlretrieve



class VLBertModel(nn.Module):
    def __init__(self, num_labels, text_pretrained='bert-base-uncased'):
        super().__init__()

        self.num_labels = num_labels
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size, num_labels)
        
    
    def forward(self, text):
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feature = text_output.last_hidden_state[:, 0, :]
        logits = self.classifier(text_feature)
        return logits


# model which extracts layers from the original ResNet-50 model
class ResNetFeatureModel(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = resnet50(pretrained=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x


# last output layer name for resnet is named 'layer4', dim 2048*7*7
# last layer name before fc is named 'avgpool', dim 2048*1*1 -> needs to be flattened
# reference: https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b

class BertResNetModel(nn.Module):
    def __init__(self, num_labels, text_pretrained='bert-base-uncased'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.visual_encoder = ResNetFeatureModel(output_layer='avgpool')
        self.image_hidden_size = 2048
        
        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.image_hidden_size, num_labels)

    def forward(self, text, image):
        text_output = self.text_encoder(**text)
        text_feature = text_output.last_hidden_state[:, 0, :]
        img_feature = self.visual_encoder(image)

        features = torch.cat((text_feature, img_feature), 1)

        logits = self.classifier(features)

        return logits



class AlbefModel(nn.Module):

    def __init__(self, bert_config, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.text_encoder = AlbefBertModel.from_pretrained(
            'bert-base-uncased', config=bert_config, add_pooling_layer=False)

        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size, num_labels)
        
    
    def forward(self, text, image):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True
                                   )
        logits = self.classifier(output.last_hidden_state[:, 0, :])
        return logits


    def load_albef_pretrained(num_out_labels):
        tmp_directory = './tmp/albef'
        os.makedirs(tmp_directory, exist_ok=True)

        albef_bert_config_fp = os.path.join(tmp_directory, 'config_bert.json')
        albef_model_fp = os.path.join(tmp_directory, 'ALBEF.pth')

        if not os.path.exists(albef_bert_config_fp):
            urlretrieve("https://raw.githubusercontent.com/salesforce/ALBEF/main/configs/config_bert.json", albef_bert_config_fp)

        if not os.path.exists(albef_model_fp):
            urlretrieve("https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth", albef_model_fp)

        albef_bert_config = AlbefBertConfig.from_json_file(albef_bert_config_fp)
        albef_model = AlbefModel(bert_config=albef_bert_config, num_labels=num_out_labels)

        albef_checkpoint = torch.load(albef_model_fp, map_location='cpu')
        albef_state_dict = albef_checkpoint['model']

        for key in list(albef_state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                albef_state_dict[encoder_key] = albef_state_dict[key]
                del albef_state_dict[key]

        msg = albef_model.load_state_dict(albef_state_dict, strict=False)
        print("ALBEF checkpoint loaded from ", albef_model_fp)
        print(msg)
        return albef_model

def create_model(image_model_type, num_labels, text_pretrained='bert-base-uncased'):
    if image_model_type is None:
        return VLBertModel(num_labels, text_pretrained=text_pretrained)
    elif image_model_type.lower().strip() == "resnet":
        return BertResNetModel(num_labels, text_pretrained=text_pretrained)
    elif image_model_type.lower().strip() == "albef":
        return AlbefModel.load_albef_pretrained(num_labels)

