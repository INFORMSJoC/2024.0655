import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x


class MMDy(nn.Module):
    def __init__(self, in_dim=(200, 512), hidden_dim= 50, num_cls=2, dropout=0.4):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(in_dim[0], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(in_dim[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.image_clf = nn.Linear(hidden_dim, num_cls)
        self.text_clf = nn.Linear(hidden_dim, num_cls)

        self.consist = Consistency(dim=hidden_dim, views=2, num_cls=num_cls)
        self.clf = nn.Linear(hidden_dim, 1)


    
    def forward(self, text, image):
        text_feature = self.text_encoder(text.sum(1))
        image_feature = self.image_encoder(image)
        text_aligned_feat, image_aligned_feat = text_feature, image_feature


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




class ConsistFusion(nn.Module):
    def __init__(self, dim=6, num_cls=2, views=2):
        super().__init__()
        self.consist = Consistency(dim=dim, views=views, num_cls=num_cls)
        self.clf = nn.Linear(dim, 1)
    

    def forward(self, datas):
        fusion_feature = self.consist(*datas.transpose(0, 1)).sum(1) # B x cls x dim
        feature = F.gelu(fusion_feature)
        logit = self.clf(feature).squeeze(-1)
        return logit



class Consistency(nn.Module):
    def __init__(self, dim=50, views=2, num_cls=2):
        super().__init__()
        self.views = views# + 1
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls, dim))
        self.QW = nn.Linear(dim, dim * self.views)
        self.FW = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.views)])
        self.FW2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.views)])
        nn.init.trunc_normal_(self.cls_token, std=0.02)


    def forward(self, *datas):
        Qs = self.QW(self.cls_token).chunk(self.views, -1)
        sims = []
        Vs = []
        for i in range(self.views):
            Q, F1, F2 = Qs[i], self.FW[i](datas[i]), self.FW2[i](datas[i])
            sim = F.cosine_similarity(Q, F1.unsqueeze(1), dim=-1) # 1 x cls x dim, B x 1 x dim -> B x cls
            V = F2.unsqueeze(1) # B x 1 x dim
            sims.append(sim)
            Vs.append(V)
        sims = torch.stack(sims, dim=1).unsqueeze(-1) # B x v x cls x 1
        Vs = torch.stack(Vs, dim=1) # B x v x 1 x dim
        F_ = (sims * Vs).sum(1)# + Vs # # B x v x cls x dim # 
        return F_, sims # B x cls x dim
    