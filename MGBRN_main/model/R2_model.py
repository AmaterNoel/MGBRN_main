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

class FFIN_C(nn.Module):
    def __init__(self, d_model=512, num_decoder_layers=1, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, tgt, mask, pos_embed):
        pos_embed = pos_embed.permute(2, 0, 1)      # b,d,t -> t,b,d

        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(query=tgt2, key=self.with_pos_embed(memory, pos_embed), value=memory, attn_mask=None, key_padding_mask=mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        return tgt

class FFIN_CC(nn.Module):
    def __init__(self, d_model=512, num_decoder_layers=1, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.Q_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.K_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.V_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, tgt, mask, pos_embed):
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        tgt2 = self.norm1(tgt)
        q = self.Q_conv(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        k = self.with_pos_embed(self.K_conv(memory.permute(1, 2, 0)).permute(2, 0, 1), pos_embed)
        v = self.V_conv(memory.permute(1, 2, 0)).permute(2, 0, 1)

        tgt2 = self.attn(query=q, key=k, value=v, attn_mask=None, key_padding_mask=mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        return tgt

class FFIN_CCL(nn.Module):
    def __init__(self, d_model=512, num_decoder_layers=1, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.Q_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.K_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.V_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, tgt, mask, pos_embed):
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        tgt2 = self.norm1(tgt)
        q = self.Q_conv(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        k = self.with_pos_embed(self.K_conv(memory.permute(1, 2, 0)).permute(2, 0, 1), pos_embed)
        v = self.V_conv(memory.permute(1, 2, 0)).permute(2, 0, 1)

        tgt2 = self.attn(query=q, key=k, value=v, attn_mask=None, key_padding_mask=mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        return tgt

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

class GroupWiseLinear_sp(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, 1, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        self.W[0][0].data.uniform_(-stdv, stdv)
        if self.bias:
            self.b[0][0].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W.expand(1, self.num_class, self.hidden_dim) * x).sum(-1)
        if self.bias:
            x = x + self.b.expand(1, self.num_class)
        return x

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

# 时空三维正余弦位置编码
class TSPositionEmbedding(nn.Module):
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

    def forward(self, mask):                                                    # mask是全false矩阵
        # print(mask.shape)                                                     # b,1569
        b, l = mask.shape
        t = (l-1)//196
        not_mask = ~mask                                                        # not_mask是全true矩阵
        not_mask = not_mask[:, 1:].reshape(b, t, 14, 14)                        # b,8,14,14

        # 时间维度上编码
        t_embed = not_mask.cumsum(1, dtype=torch.float32).to(self.device)       # b,8,14,14
        # 序列编码是一维的序列进行embed，特征图是二维的，编码时分别做x方向的累积和y方向的累加
        y_embed = not_mask.cumsum(2, dtype=torch.float32).to(self.device)       # b,8,14,14
        # print(y_embed.shape)
        x_embed = not_mask.cumsum(3, dtype=torch.float32).to(self.device)       # b,8,14,14
        # print(x_embed.shape)
        if self.normalize:
            # 防止除以0的操作
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 制作一个256维的向量，并将这256维的向量区分为奇数和偶数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32).to(self.device)           # 256 ps:从0到256的一维序列
        # print(dim_t.shape)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)                     # 256 ps:映射到1到10000

        pos_t = t_embed[:, :, :, :, None] / dim_t           # 8,8,14,14,256     ps:时间维度256
        pos_x = x_embed[:, :, :, :, None] / dim_t           # 8,8,14,14,256     ps:空间维度256
        pos_y = y_embed[:, :, :, :, None] / dim_t           # 8,8,14,14,256     ps:空间维度256
        pos_t = torch.stack((pos_t[:, :, :, :, 0::2].sin(), pos_t[:, :, :, :, 1::2].cos()), dim=5).flatten(4)     # 8,8,14,14,256
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)     # 8,8,14,14,256
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)     # 8,8,14,14,256
        pos_xy = torch.cat((pos_y, pos_x), dim=4)
        pos = torch.cat((pos_t, pos_xy), dim=4).reshape(b, -1, self.num_pos_feats * 3)      # 8,1568,768
        cls_tokens = torch.zeros(b, 1, self.num_pos_feats * 3).to(self.device)                  # 8,1,768
        pos = torch.cat((cls_tokens, pos), dim=1)                                           # 8,1569,768
        pos = pos.unsqueeze(1).permute(0, 3, 1, 2)                                              # 8,768,1,1569
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

class CFNm(nn.Module):
    def __init__(self, RT_model, TST_model, num_queries, device):
        super(CFNm, self).__init__()
        self.device = device
        self.embed_dim = 768
        self.num_action_class = num_queries
        self.RT_model = RT_model
        self.TST_model = TST_model

        self.ffin = FFIN_C(d_model=self.embed_dim, dropout=0.1, nhead=8, dim_feedforward=2048, normalize_before=True, return_intermediate_dec=True)

        self.decoder = Decoder(embed_dim=self.embed_dim, dropout=0.1, nhead=8, dim_feedforward=2048)
        self.query_embed = nn.Embedding(self.num_action_class, self.embed_dim)

        self.RT_position_embedding = PositionEmbeddingSine(self.embed_dim, normalize=False, device=self.device)
        self.TST_position_embedding = PositionEmbeddingSine(self.embed_dim, normalize=False, device=self.device)

        self.linear = nn.Linear(1569, self.num_action_class)
        self.action_group_linear = GroupWiseLinear(self.num_action_class, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        RT_out, RT_pred = self.RT_model(images)         # b,t,d
        TST_out, TST_pred = self.TST_model(images)      # b,t,d


        RT_out = RT_out.permute(1, 0, 2)                                        # t,b,d
        RT_mask = torch.zeros((b, t), dtype=torch.bool).to(self.device)
        RT_pos = self.RT_position_embedding(RT_mask)                            # b,d,t (8,768,8)

        TST_out = TST_out.permute(1, 0, 2)                                      # b,t,d -> t,b,d
        hs = self.ffin(RT_out, TST_out, RT_mask, RT_pos)                        # t,8,768

        # 补充detr-decoder
        TST_mask = torch.zeros((b, hs.shape[0]), dtype=torch.bool).to(self.device)  # 8,t
        TST_pos = self.TST_position_embedding(TST_mask).unsqueeze(2)                # 8,768,1,t
        hs = self.decoder(hs, self.query_embed.weight, TST_mask, TST_pos)[-1]       # 8,140,768

        # MLP部分
        #hs = hs.permute(1, 2, 0)  # b,d,1569
        #hs = self.linear(hs)  # b,d,140
        #hs = hs.permute(0, 2, 1)  # b,140,d

        hs = self.action_group_linear(hs)
        return hs