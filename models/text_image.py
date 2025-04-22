import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import transformers

from .mmdy import Consistency
from .Encoder import TransformerEncoder
from .vit import VisionTransformer

import timm


def image_patch(iamges, img_size=224, patch_size=16, in_c=3):
    images = iamges.unfold(-2, patch_size, patch_size).unfold(-2, patch_size, patch_size).contiguous() \
        .view(-1, in_c, (img_size // patch_size) ** 2, patch_size*patch_size).transpose(-2, -3).contiguous() \
        .view(-1, (img_size // patch_size) ** 2, patch_size*patch_size*in_c)
    return images # B x 192 x 768

# torchvision.models.vit_b_16
class VitImageEncoder(nn.Module):
    def __init__(self, weights="IMAGENET1K_SWAG_E2E_V1"):
        super(VitImageEncoder, self).__init__()

        self.model = VisionTransformer(image_size=384,
                                    patch_size=16,
                                    num_layers=12,
                                    num_heads=12,
                                    hidden_dim=768,
                                    mlp_dim=3072, )
        weights = torchvision.models.ViT_B_16_Weights.verify(weights)
        self.model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))


    def forward(self, x):
        x = self.model._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        out = x[:, 0]

        # out = self.model(x)
        return out
    

class TextImage_weibo(nn.Module):
    def __init__(self, in_dim=(768, 768), hidden_dim= 128, num_cls=2, dropout=0.4):
        super().__init__()
        self.text_encoder = transformers.BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", local_files_only=True)
        self.image_encoder = VitImageEncoder()
        
        self.text_proj = nn.Linear(in_dim[0], hidden_dim)
        self.image_proj = nn.Linear(in_dim[1], hidden_dim)
        self.image_clf = nn.Linear(hidden_dim, num_cls)
        self.text_clf = nn.Linear(hidden_dim, num_cls)
        self.consist = Consistency(dim=hidden_dim, views=2, num_cls=num_cls)
        self.clf = nn.Linear(hidden_dim, 1)
    

    def forward(self, token_ids, attn_mask, image):
        text_feature = self.text_encoder(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state[:, 0]
        image_feature = self.image_encoder(image)

        text_aligned_feat, image_aligned_feat = self.text_proj(text_feature), self.image_proj(image_feature)

        text_logit, image_logit = self.text_clf(text_aligned_feat), self.image_clf(image_aligned_feat)

        fusion_feature, sims = self.consist(text_aligned_feat, image_aligned_feat)#.sum(1) # B x cls x dim
        feature = F.gelu(fusion_feature)
        logit = self.clf(feature).squeeze(-1)


        return_dict = {
            "text_feat": text_feature,
            "image_feat": image_feature,
            "text_aligned": text_aligned_feat,
            "image_aligned": image_aligned_feat,
            "feat": fusion_feature,
            "logits": logit,
            "text_logit": text_logit, 
            "image_logit": image_logit,
            "sims": sims,
        }
        return logit, return_dict


class TextImage(nn.Module):
    def __init__(self, in_dim=(768, 768), hidden_dim= 128, num_cls=2, dropout=0.4):
        super().__init__()
        self.text_encoder = transformers.RobertaModel.from_pretrained("FacebookAI/roberta-base", local_files_only=True)
        self.image_encoder = VitImageEncoder()

        self.text_proj = nn.Linear(in_dim[0], hidden_dim)
        self.image_proj = nn.Linear(in_dim[1], hidden_dim)
        self.image_clf = nn.Linear(hidden_dim, num_cls)
        self.text_clf = nn.Linear(hidden_dim, num_cls)
        self.consist = Consistency(dim=hidden_dim, views=2, num_cls=num_cls)
        self.clf = nn.Linear(hidden_dim, 1)
    

    def forward(self, token_ids, attn_mask, image):
        text_feature = self.text_encoder(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state[:, 0]
        image_feature = self.image_encoder(image)

        text_aligned_feat, image_aligned_feat = self.text_proj(text_feature), self.image_proj(image_feature)

        text_logit, image_logit = self.text_clf(text_aligned_feat), self.image_clf(image_aligned_feat)

        fusion_feature, sims = self.consist(text_aligned_feat, image_aligned_feat)#.sum(1) # B x cls x dim
        feature = F.gelu(fusion_feature)
        logit = self.clf(feature).squeeze(-1)

        return_dict = {
            "text_feat": text_feature,
            "image_feat": image_feature,
            "text_aligned": text_aligned_feat,
            "image_aligned": image_aligned_feat,
            "feat": fusion_feature,
            "logits": logit,
            "text_logit": text_logit, 
            "image_logit": image_logit,
            "sims": sims,
        }
        return logit, return_dict


