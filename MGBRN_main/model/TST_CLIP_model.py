import timm
import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from transformers import TimesformerModel, TimesformerForVideoClassification

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

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

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
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # tgt为初始化为0的query向量
        q = k = self.with_pos_embed(tgt, query_pos)
        # print(q.shape)

        # 使用的是torch提供的nn.MultiheadAttention，与self.multihead_attn使用的是同一个模块，但是参数值各自训练
        # tgt的100个查询向量全使用，所以key_padding_mask=None
        # decoder的自注意力模块
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # decoder的多头注意力模块，q由decoder的自注意力模块提供，k,v由encoder的memory提供
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # 残差连接
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN(Feed Forward Network): norm+linear+relu+linear+norm
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # 残差连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
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

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # 默认self.normalize_before=False，只执行forward_post
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class FFIN(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, query_embed, mask, pos_embed):
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
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
        # print(mask.shape)      # b,h,w
        not_mask = ~mask

        # 序列编码是一维的序列进行embed，特征图是二维的，编码时分别做x方向的累积和y方向的累加
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(self.device)
        # print(y_embed.shape)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(self.device)
        # print(x_embed.shape)
        if self.normalize:
            # 防止除以0的操作
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 制作一个128维的向量，并将这128维的向量区分为奇数和偶数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32).to(self.device)
        # print(dim_t.shape)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TST_CLIP(nn.Module):
    def __init__(self, text_embed, num_frames, num_queries, device):
        super(TST_CLIP, self).__init__()
        self.text_embed = text_embed.to(device)
        self.frames = num_frames
        self.device = device
        self.embed_dim = 256
        self.num_action_class = num_queries

        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                         num_frames=self.frames, ignore_mismatched_sizes=True)
        self.ffin = FFIN(d_model=256, dropout=0.1, nhead=8, dim_feedforward=2048, normalize_before=True, return_intermediate_dec=True)

        self.position_embedding = PositionEmbeddingSine(self.embed_dim // 2, normalize=True, device=self.device)
        self.linear = nn.Linear(self.backbone.config.hidden_size, self.embed_dim)
        self.action_group_linear = GroupWiseLinear(self.num_action_class, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)[0]
        x = self.linear(x)
        s_len = x.shape[1]
        x = x.permute(1, 0, 2)

        mask = torch.zeros((b, 1, s_len), dtype=torch.bool).to(self.device)
        pos = self.position_embedding(mask)

        query_embed = self.text_embed.unsqueeze(1).repeat(1, b, 1)
        hs = self.ffin(x, query_embed, mask, pos)[-1]
        x = self.action_group_linear(hs)
        return x


class TST_CLIP_NH(nn.Module):
    def __init__(self, text_embed, num_frames, num_queries, device):
        super(TST_CLIP_NH, self).__init__()
        self.text_embed = text_embed.to(device)
        self.frames = num_frames
        self.device = device
        self.embed_dim = 256
        self.num_action_class = num_queries

        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                         num_frames=self.frames, ignore_mismatched_sizes=True)
        self.ffin = FFIN(d_model=256, dropout=0.1, nhead=8, dim_feedforward=2048, normalize_before=True, return_intermediate_dec=True)

        self.position_embedding = PositionEmbeddingSine(self.embed_dim // 2, normalize=True, device=self.device)
        self.linear = nn.Linear(self.backbone.config.hidden_size, self.embed_dim)
        self.action_group_linear = GroupWiseLinear(self.num_action_class, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)[0]
        x = self.linear(x)
        s_len = x.shape[1]
        x = x.permute(1, 0, 2)

        mask = torch.zeros((b, 1, s_len), dtype=torch.bool).to(self.device)
        pos = self.position_embedding(mask)

        query_embed = self.text_embed.unsqueeze(1).repeat(1, b, 1)
        hs = self.ffin(x, query_embed, mask, pos)[-1]
        return hs