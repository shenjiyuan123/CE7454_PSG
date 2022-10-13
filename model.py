# -------------------------------------------------------------------------
# Copyright (c) 2022 lucidrains/vit-pytorch
#
# Modified by Jiyuan SHEN
# -------------------------------------------------------------------------
import torch
from torch import nn
import clip

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *tmp):
        if tmp != ():
            return self.fn(self.norm(x), tmp[0])
        else:
            return self.fn(self.norm(x))

class EasyFF(nn.Module):
    def __init__(self, in_feature, out_feature, dropout=0.):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_feature,out_feature),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.proj(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        assert dim == inner_dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        assert dim == inner_dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, cross_kv):
        kv = self.to_kv(cross_kv).chunk(2, dim=-1)
        kk, vv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        qq = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(qq, kk.transpose(-1,-2))*self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vv)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_layers(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,dim_head=dim_head,dropout=dropout)),
                PreNorm(dim, FeedForward(dim,hidden_dim=mlp_dim,dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Grouping_blocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim,heads=heads,dim_head=dim_head,dropout=dropout)),
                PreNorm(dim, FeedForward(dim,hidden_dim=mlp_dim,dropout=dropout))
            ]))

    def forward(self, q, cross_kv):
        for cro_attn, ff in self.layers:
            org = q
            mid = cro_attn(q, cross_kv)
            mid = ff(mid)
            out = org + mid
        return out

class text_embedding(nn.Module):
    def __init__(self,vocab_size,dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.token_embedding(x)

def clip_tokenize():
    """use the clip text encoder to tokenize the class labels

    Returns:
        tensor: shape likes [56,512]
    """
    predicate_classes= ["over", "in front of", "beside", "on", "in", "attaching", "hanging", "on back of", "falling", "going", "painted", "walking", "running", "crossing", "standing", "lying", "sitting", "flying", "jumping", "jumping", "wearing", "holding", "carrying", "looking", "guiding", "kissing", "eating", "drinking", "feeding", "biting", "catching", "picking", "playing", "chasing", "climbing", "cleaning", "playing", "touching", "pushing", "pulling", "opening", "cooking", "talking", "throwing", "slicing", "driving", "riding", "parked", "driving", "hitting", "kicking", "swinging", "entering", "exiting", "enclosing", "leaning"]
    # predicate_classes= ["over", "in front of", "beside", "on", "in", "attached to", "hanging from", "on back of", "falling off", "going down", "painted on", "walking on", "running on", "crossing", "standing on", "lying on", "sitting on", "flying over", "jumping over", "jumping from", "wearing", "holding", "carrying", "looking at", "guiding", "kissing", "eating", "drinking", "feeding", "biting", "catching", "picking", "playing with", "chasing", "climbing", "cleaning", "playing", "touching", "pushing", "pulling", "opening", "cooking", "talking to", "throwing", "slicing", "driving", "riding", "parked on", "driving on", "about to hit", "kicking", "swinging", "entering", "exiting", "enclosing", "leaning on"]
    model, _ = clip.load("ViT-B/16")
    text = torch.cat([clip.tokenize(f"action of {c}") for c in predicate_classes]).cuda()
    token = model.encode_text(text)
    return token

class Group_vit(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth_T, depth_G, heads, mlp_dim, vocab_size, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.proj_img = EasyFF(in_feature=1024,out_feature=dim,dropout=dropout) # 768 for the base clip, 1024 for the largest
        self.proj_text = EasyFF(in_feature=512,out_feature=dim,dropout=dropout) # 512 for the clip word tokenization

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # for the clip extraction feas pos_embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 577, dim))  #197 for the base clip, 577 for the largest
        # self.text_token = text_embedding(vocab_size,dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer_layers(dim,depth_T,heads,dim_head,mlp_dim,dropout)
        self.groupblock1  = Grouping_blocks(dim,depth_G,heads,dim_head,mlp_dim,dropout)

        self.cls_token = nn.Parameter(torch.randn(1,1,dim))

        self.transformer2 = Transformer_layers(dim,depth_T,heads,dim_head,mlp_dim,dropout)
        self.groupblock2  = Grouping_blocks(dim,depth_G,heads,dim_head,mlp_dim,dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1) #output shape=[N,C,L_output]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.pos_embedding, std=0.01)
        # nn.init.normal_(self.proj_img.weight, std=0.01)
        # nn.init.normal_(self.proj_text.weight, std=0.01)

        std = 512 ** -0.5
        for att,ff in self.transformer1.layers:
            nn.init.normal_(att.fn.to_qkv.weight, std=std)
            nn.init.normal_(att.fn.to_out[0].weight, std=std)
            nn.init.normal_(ff.fn.net[0].weight, std=std)
            nn.init.normal_(ff.fn.net[3].weight, std=std)
        for cro,ff in self.groupblock1.layers:
            nn.init.normal_(cro.fn.to_kv.weight, std=std)
            nn.init.normal_(cro.fn.to_out[0].weight, std=std)
            nn.init.normal_(ff.fn.net[0].weight, std=std)
            nn.init.normal_(ff.fn.net[3].weight, std=std)
        for att,ff in self.transformer2.layers:
            nn.init.normal_(att.fn.to_qkv.weight, std=std)
            nn.init.normal_(att.fn.to_out[0].weight, std=std)
            nn.init.normal_(ff.fn.net[0].weight, std=std)
            nn.init.normal_(ff.fn.net[3].weight, std=std)
        for cro,ff in self.groupblock2.layers:
            nn.init.normal_(cro.fn.to_kv.weight, std=std)
            nn.init.normal_(cro.fn.to_out[0].weight, std=std)
            nn.init.normal_(ff.fn.net[0].weight, std=std)
            nn.init.normal_(ff.fn.net[3].weight, std=std)


    def forward(self, x, text):
        # x = self.to_patch_embedding(img)
        x = self.proj_img(x)
        bs, n, _ = x.shape
        x += self.pos_embedding

        # y_tmp = self.text_token(text).unsqueeze(0)
        y_tmp = text.unsqueeze(0)
        
        y = repeat(y_tmp, '1 n d -> b n d', b=bs)
        y = self.proj_text(y)

        xy = torch.cat((x,y), dim=1)

        xy = self.transformer1(xy)
        xx = xy[:,:n]
        yy = xy[:,n:]
        xy = self.groupblock1(xx,yy)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=bs)
        xy = torch.cat((xy,cls_tokens), dim=1)

        xy = self.transformer2(xy)
        # xx = xy[:,:n]
        # yy = xy[:,n:]
        # out = self.groupblock2(yy,xx)
        out = xy[:,n,:]

        # out = self.avgpool(out.transpose(1,2))
        out = torch.flatten(out,1)
        pred = self.mlp_head(out)

        return pred


'''
v = Group_vit(
    image_size = 224,
    patch_size = 16,
    num_classes = 50,
    dim = 512,
    depth_T = 6,
    depth_G = 1,
    heads = 8,
    mlp_dim = 1024,
    vocab_size=100,
    dim_head=64,
    dropout = 0.1,
    emb_dropout = 0.1
).cuda()


img = torch.randn(10, 197, 768).cuda()
# text = torch.randint(0,49,(50,))
text = clip_tokenize().cuda()
preds = v(img,text) 
print(preds.shape)

# print(v)
print(sum(p.numel() for p in v.parameters() if p.requires_grad))
'''
