# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer
import torch
class AttentionSelector(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionSelector, self).__init__()
        self.dense = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, hidden_states):
        selector_tensor = self.dense(hidden_states).squeeze(-1)
        # return torch.sigmoid(selector_tensor)
        return torch.softmax(selector_tensor, dim=1)

class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        self.selector = nn.ModuleList([AttentionSelector(args.hidden_size) for _ in range(self.layers_num)])

    def forward(self, pre, emb, seg, vm=None, dep=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if vm is None:
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vm.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        if dep is None:
            dep_mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            dep_mask = dep_mask.float()
            dep_mask = (1.0 - dep_mask) * -10000.0
        else:
            dep_mask = dep.unsqueeze(1)
            dep_mask = dep_mask.float()
            dep_mask = (1.0 - dep_mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            selector_outputs = self.selector[i](hidden)[:, None, :, None]
            hidden = self.transformer[i](pre, hidden, mask, dep_mask, selector_outputs)
            pre = hidden
        return hidden

