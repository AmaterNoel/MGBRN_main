import timm
import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# encoder架构
class Encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        b, d, t = src.shape                             # 8,768,8
        # 将输入数据拉长成序列，序列长度为 24*36，维度是256
        src = src.permute(2, 0, 1)                      # t,b,d=8,8,768
        # print(src.shape)
        # pos和src是一样的结构，所以处理也完全一样
        pos_embed = pos_embed.permute(2, 0, 1)          # t,b,d=8,8,768

        # DETR中encoder，输入包括特征序列src，mask指明了序列当中哪些是padding的，位置编码pos_embed
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        # 将多头注意力机制复制6层
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 只对q和k加入了位置编码，对v并不加上位置编码
        q = k = self.with_pos_embed(src, pos)
        # print(q.shape)

        # 使用的是torch提供的nn.MultiheadAttention，返回值有两个，一个是计算完成的特征图，一个是权重项(用于可视化)，目标检测任务只需要特征图，所以取[0]
        # nlp中需要将decoder中的目标序列mask掉，实现逐词预测，需要src_mask，物体检测可以同时做
        # key_padding_mask指明了序列的哪些位置与任务无关，不需要计算
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # print(src2.shape)

        # 残差连接
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN(Feed Forward Network): norm+linear+relu+linear+norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # 残差连接
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

# RT标准正余弦位置编码
class RTPositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, device=torch.device("cpu")):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # print(mask.shape)      # b,t
        not_mask = ~mask

        # 序列编码是一维的序列进行embed
        x_embed = not_mask.cumsum(1, dtype=torch.float32).to(self.device)               # 8,8
        # print(x_embed.shape)
        if self.normalize:
            # 防止除以0的操作
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 制作一个128维的向量，并将这128维的向量区分为奇数和偶数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32).to(self.device)   # 768
        # print(dim_t.shape)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t                                             # 8,8,768
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)                                                    # b,t,d -> b,d,t
        return pos

# 标准正余弦位置编码
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, device=torch.device("cpu")):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # print(mask.shape)      # b,t
        not_mask = ~mask

        # 序列编码是一维的序列进行embed
        t_embed = not_mask.cumsum(1, dtype=torch.float32).to(self.device)
        if self.normalize:
            # 防止除以0的操作
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, -1:] + eps) * self.scale

        # 制作一个768维的向量，并将这768维的向量区分为奇数和偶数
        dim_t = torch.arange(2 * self.num_pos_feats, dtype=torch.float32).to(self.device)
        # print(dim_t.shape)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (2 * self.num_pos_feats))

        pos_t = t_embed[:, :, None] / dim_t
        pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_t.permute(0, 2, 1)
        return pos

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        # 将多头注意力机制复制6层
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            intermediate.pop()
            intermediate.append(output)

        return torch.stack(intermediate)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class Decoder(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(embed_dim, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.embed_dim = embed_dim
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, query_embed, mask, pos_embed):
        bs = memory.shape[1]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)       # 1569,8,768
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # 140,8,768
        tgt = torch.zeros_like(query_embed)                     # 140,8,768
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class RTwithoutFFIN(nn.Module):
    def __init__(self, num_queries, frames, device):
        super(RTwithoutFFIN, self).__init__()
        self.frames = frames
        self.device = device
        self.embed_dim = 768
        self.num_action_class = num_queries

        self.backbone = timm.create_model('resnetv2_50', pretrained=True, num_classes=self.embed_dim)

        self.position_embedding = PositionEmbeddingSine(self.embed_dim // 2, normalize=False, device=self.device)
        self.encoder = Encoder(d_model=self.embed_dim, dropout=0.1, nhead=8, dim_feedforward=2048, normalize_before=True,
                               return_intermediate_dec=True)

        self.RT_position_embedding = RTPositionEmbeddingSine(self.embed_dim, normalize=False, device=self.device)
        self.decoder = Decoder(embed_dim=self.embed_dim, dropout=0.1, nhead=8, dim_feedforward=2048)
        self.query_embed = nn.Embedding(self.num_action_class, self.embed_dim)

        self.linear = nn.Linear(self.frames * self.embed_dim, self.num_action_class)
        self.action_group_linear = GroupWiseLinear(self.num_action_class, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        images = images.reshape(-1, c, h, w)
        x = self.backbone(images)                                       # b*16,768
        x = x.reshape(b, t, -1).permute(0, 2, 1)                        # b,768,t

        mask = torch.zeros((b, t), dtype=torch.bool).to(self.device)    # b,t
        pos = self.position_embedding(mask)                             # b,768,t

        x = self.encoder(x, mask, pos)                                 # t,b,768
        hs = x.permute(1, 0, 2)                                        # t,b,d -> b,t,d

        # 补充detr-decoder
        #RT_mask = torch.zeros((b, x.shape[0]), dtype=torch.bool).to(self.device)   # 8,8
        #RT_pos = self.RT_position_embedding(RT_mask)                                # 8,768,1,8
        #x = self.decoder(x, self.query_embed.weight, RT_mask, RT_pos)[-1]          # 8,140,768
        #x = self.action_group_linear(x)

        # MLP
        x = hs.reshape(b, -1)
        x = self.linear(x)
        return hs, x

class RTwithoutFFIN_NH(nn.Module):
    def __init__(self, num_queries, frames, device):
        super(RTwithoutFFIN_NH, self).__init__()
        self.frames = frames
        self.device = device
        self.embed_dim = 256
        self.num_action_class = num_queries

        self.backbone = timm.create_model('resnetv2_50', pretrained=True, num_classes=256)

        self.position_embedding = PositionEmbeddingSine(self.embed_dim // 2, normalize=True, device=self.device)
        self.encoder = Encoder(d_model=256, dropout=0.1, nhead=8, dim_feedforward=2048, normalize_before=True,
                               return_intermediate_dec=True)

        self.action_group_linear = nn.Linear(self.embed_dim * self.frames, self.num_action_class)

    def forward(self, images):
        b, t, c, h, w = images.size()           # 1,16,3,224,224
        images = images.reshape(-1, c, h, w)    # 16,3,224,224
        x = self.backbone(images)               #
        x = x.reshape(b, t, -1)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)

        mask = torch.zeros((b, 1, t), dtype=torch.bool).to(self.device)
        pos = self.position_embedding(mask)

        hs = self.encoder(x, mask, pos)
        hs = hs.permute(1, 0, 2)

        return hs