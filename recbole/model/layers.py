
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_

from recbole.utils import FeatureType, FeatureSource


class MLPLayers(nn.Module):
    r""" MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0., activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name='relu', emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'dice':
            activation = Dice(emb_dim)
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class FMEmbedding(nn.Module):
    r""" Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x ** 2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_signal = fn.relu(att_signal)  # [batch_size, M, att_dim]

        att_signal = torch.mul(att_signal, self.h)  # [batch_size, M, att_dim]
        att_signal = torch.sum(att_signal, dim=2)  # [batch_size, M]
        att_signal = fn.softmax(att_signal, dim=1)  # [batch_size, M]

        return att_signal


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self, mask_mat, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=False, return_seq_weight=True
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(self.att_hidden_size, activation='Sigmoid', bn=False)
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = (mask >= keys_length.unsqueeze(1))

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size ** 0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):  # max len, max len
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):              # for dif-sr, VanillaAttention input_tensor: [B,h,L,fea_num+2,L]   [B,h,L,(fea num+2)**2,L]
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)                  # [B,h,L,fea_num+2,1]    [B, ]
        weights = torch.softmax(energy.squeeze(-1), dim=-1)     # [B,h,L,fea_num+2]
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        # [B,h,L,fea_num+2,L] * [B,h,L,fea_num+2,1] = [B,h,L,fea_num+2,L] sum dim=-2-> [B,h,L,L]
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)  # the only value layer

        self.query_p = nn.Linear(hidden_size, self.all_head_size)      # position query
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)   # 2+self.feat_num  2:item,position
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):        # partition of each attention head
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [batch size, head num, seq len, emb dim]
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x, i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))   # [B,h,L,L]
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))   # [B,h,L,L]

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)    # here is the cate emb [bs, L, 1, d_f] -> [bs, L, d_f]
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor), i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor), i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))  # [B,h,L,L]
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))   # append [B,h,L,1,L]

        attribute_attention_table = torch.cat(attribute_attention_table, dim=-2)  # after cat [B,h,L,1,L]
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]            # feat_atten_num, attention_size = 1, L
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)   # [B,h,L,1,L] -> [B,h,L,L]
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))  # [B,h,L,fea num,L]->[B,h,L,fea num*L]
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)  # [B,h,L,(fea_num+2)*L]
            attention_scores = self.fusion_layer(attention_scores)  # [B,h,L,(fea_num+2)*L] -> [B,h,L,L]
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat([attribute_attention_table, item_attention_scores.unsqueeze(-2),
                                          pos_scores.unsqueeze(-2)], dim=-2)    # get [B,h,L,fea_num+2,L]
            attention_scores, _ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()               # [B,h,L,d/h] -> [B,L,h,d/h]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [B,L,d]
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)        # linear projection after the concatenation of multi head attention
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(CrossMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)  # the only value layer

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(8+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        w_matrix = torch.empty(3, 3)
        self.w_matrix = torch.nn.init.xavier_normal_(w_matrix)
        self.w_matrix = nn.Parameter(self.w_matrix)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        # [bs, h, L, d/h]
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))
        # [bs, h, L, d/h]
        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))
        # [bs, h, L, L]
        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))  # 1
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))               # 2

        # [bs, h, L, L]
        i_p_attention_scores = torch.matmul(item_query_layer, pos_key_layer.transpose(-1, -2))    # 3
        p_i_attention_scores = torch.matmul(pos_query_layer, item_key_layer.transpose(-1, -2))    # 4

        attribute_attention_table = []
        i = 0
        attribute_query, attribute_key = self.query_layers[i], self.key_layers[i]
        # [bs, L, d]
        attribute_tensor = attribute_table[i].squeeze(-2)
        # [bs, h, L, d/h]
        attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor), i)
        attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor), i)
        # [bs, h, L, L]
        attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))   # 5
        attribute_i_attention_scores = torch.matmul(attribute_query_layer, item_key_layer.transpose(-1, -2))   # 6
        attribute_p_attention_scores = torch.matmul(attribute_query_layer, pos_key_layer.transpose(-1, -2))   # 7
        i_attribute_attention_scores = torch.matmul(item_query_layer, attribute_key_layer.transpose(-1, -2))   # 8
        p_attribute_attention_scores = torch.matmul(pos_query_layer, attribute_key_layer.transpose(-1, -2))   # 9

        item_attention_scores = item_attention_scores * self.w_matrix[0][0]
        i_p_attention_scores = i_p_attention_scores * self.w_matrix[0][1]
        i_attribute_attention_scores = i_attribute_attention_scores * self.w_matrix[0][2]
        p_i_attention_scores = p_i_attention_scores * self.w_matrix[1][0]
        pos_scores = pos_scores * self.w_matrix[1][1]
        p_attribute_attention_scores = p_attribute_attention_scores * self.w_matrix[1][2]
        attribute_i_attention_scores = attribute_i_attention_scores * self.w_matrix[2][0]
        attribute_p_attention_scores = attribute_p_attention_scores * self.w_matrix[2][1]
        attribute_attention_scores = attribute_attention_scores * self.w_matrix[2][2]

        # [bs, h, L, num_feature, L] num_feature = 1
        attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))

        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape  # [bs, h, L, num_feature, L] num_feature = 1
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores + attribute_i_attention_scores +\
                               attribute_p_attention_scores + i_attribute_attention_scores + p_attribute_attention_scores +\
                               i_p_attention_scores + p_i_attention_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat(
                [attention_scores, item_attention_scores, pos_scores, attribute_i_attention_scores,
                 attribute_p_attention_scores, i_attribute_attention_scores, p_attribute_attention_scores,
                 i_p_attention_scores, p_i_attention_scores], dim=-1)  # [bs, h, L, L*N]
            attention_scores = self.fusion_layer(attention_scores)  # [bs, h, L, L]
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2),
                 attribute_i_attention_scores.unsqueeze(-2),
                 attribute_p_attention_scores.unsqueeze(-2), i_attribute_attention_scores.unsqueeze(-2),
                 p_attribute_attention_scores.unsqueeze(-2),
                 i_p_attention_scores.unsqueeze(-2), p_i_attention_scores.unsqueeze(-2)], dim=-2)
            attention_scores, _ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ACMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(ACMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)  # value projection layer for the field of "item id"

        self.query_p = nn.Linear(hidden_size, self.all_head_size)      # position query
        self.key_p = nn.Linear(hidden_size, self.all_head_size)
        self.value_p = nn.Linear(hidden_size, self.all_head_size)  # value   for position

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.value_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])          # value   for all the side info

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len * ((2+self.feat_num)**2), self.max_len)   # (2+self.feat_num)**2  2:item,position
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)


    def transpose_for_scores(self, x):        # partition of each attention head
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [batch size, head num, seq len, emb dim]
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x, i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))
        pos_value_layer = self.transpose_for_scores(self.value_p(position_embedding))

        attribute_query_layers, attribute_key_layers, attribute_value_layers = [], [], []
        for i, (attribute_query, attribute_key, attribute_value) in enumerate(zip(self.query_layers, self.key_layers, self.value_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)     # here is the cate emb [bs, L, 1, d_f] -> [bs, L, d_f]
            attribute_query_layers.append(self.transpose_for_scores_attribute(attribute_query(attribute_tensor), i))
            attribute_key_layers.append(self.transpose_for_scores_attribute(attribute_key(attribute_tensor), i).unsqueeze(1))    # [B,1,H,L,D] list
            attribute_value_layers.append(self.transpose_for_scores_attribute(attribute_value(attribute_tensor), i).unsqueeze(1))  # [B,1,H,L,D] LIST

        # queries  [[B,H,L,D], [B,H,L,D], [B,H,L,D]]  [item_query, pos_query, attribute_query]
        queries = []
        queries.append(item_query_layer)
        queries.append(pos_query_layer)
        queries = queries + attribute_query_layers     # list length: 1+1+side info field num   F+1

        # keys: tensor [B,F+1,H,L,D]    i [c] p  F+1            [B,F+1,H,L,D]
        keys = torch.cat([item_key_layer.unsqueeze(1), torch.cat(attribute_key_layers, dim=1), pos_key_layer.unsqueeze(1)], dim=1)
        # value: [B,F,H,L,D]  i [c]   TODO: whether the position info is considered
        values = torch.cat([item_value_layer.unsqueeze(1), torch.cat(attribute_value_layers, dim=1)], dim=1)

        # all the queries cross all the keys
        raw_attention = []
        for i in range(len(queries)):
            query = queries[i].unsqueeze(1).expand(-1, keys.shape[1], -1, -1, -1)  # [B,H,L,D] -> [B,1,H,L,D] -> [B,F+1,H,L,D]
            attention_temp = torch.matmul(query, keys.transpose(-1, -2))           # [B,F+1,H,L,L]
            raw_attention.append(attention_temp)

        ac_raw_attention = torch.cat(raw_attention, dim=1)    # [B,(F+1)**2,H,L,L]
        # currently, (F+1)**2 types of attention combinations.  only id info is available in value matrices
        # (F+1)**2 types of raw attention score.  sum or concat or gate?
        ac_attention = torch.permute(ac_raw_attention, (0, 2, 3, 1, 4))      # [B,(F+1)**2,H,L,L] -> # [B,H,L,(F+1)**2,L]
        if self.fusion_type == 'sum':
            ac_attention = torch.sum(ac_raw_attention, dim=-2)   # [B,H,L,(F+1)**2,L] -> [B,H,L,L]
        elif self.fusion_type == 'concat':
            ac_attn_shape = ac_attention.shape
            cross_num, attn_size = ac_attn_shape[-2], ac_attn_shape[-1]
            ac_attention = ac_attention.view(ac_attn_shape[:-2] + (cross_num * attn_size, ))   #  [B,H,L,(F+1)**2,L]->[B,H,L,(F+1)**2 *L]
            ac_attention = self.fusion_layer(ac_attention)      # [B,H,L,(F+1)**2 *L] -> [B,H,L,L]
        elif self.fusion_type == 'gate':
            # VanillaAttention input_tensor: [B,h,L,(fea_num+2)**2,L] or rather: [B,h,L,(F+1)**2,L]   -> [B,h,L,L]
            ac_attention, _  = self.fusion_layer(ac_attention)

        ac_attention = ac_attention / math.sqrt(self.attention_head_size)

        ac_attention = ac_attention + attention_mask
        attention_probs = nn.Softmax(dim=-1)(ac_attention)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)      # all-cross  attention  only item id value matrices
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B,h,L,d/h] -> [B,L,h,d/h]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [B,L,d]
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)  # linear projection after the concatenation of multi head attention
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class CrossTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(CrossTransformerLayer, self).__init__()
        self.multi_head_attention = CrossMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class ACTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(ACTransformerLayer, self).__init__()
        self.multi_head_attention = ACMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class CrossTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(CrossTransformerEncoder, self).__init__()
        layer = CrossTransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ACTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(ACTransformerEncoder, self).__init__()
        layer = ACTransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states, position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states, position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.token_embedding_table = {}
        self.float_embedding_table = {}
        self.token_seq_embedding_table = {}

        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field.

        """
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}

        for type in self.types:
            # print(self.field_names[type])
            for field_name in self.field_names[type]:
                # print('===========', field_name)
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                    # print('layers.py: TOKEN')
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    # print('==================== \n layers.py: field name:', field_name)
                    self.token_seq_field_names[type].append(field_name)
                    # print('==================== \n layers.py: dataset.num(field_name):', self.dataset.num(field_name))    
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    # print('layers.py else')
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features.

        """
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                # print('line 1305')
                self.token_field_offsets[type] = np.array((0, *np.cumsum(self.token_field_dims[type])[:-1]),
                                                          dtype=np.long)
                self.token_embedding_table[type] = FMEmbedding(
                    self.token_field_dims[type], self.token_field_offsets[type], self.embedding_size
                ).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                # print('line 1318')
                self.float_embedding_table[type] = nn.Embedding(
                    np.sum(self.float_field_dims[type], dtype=np.int32), self.embedding_size
                ).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    # print('======================== \n token_seq_field_dim: ', token_seq_field_dim)    # 类别数
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(self.device)
                    )

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[-1]
        # [batch_size, max_item_length, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, max_item_length, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table[type](index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(-1))

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == 'item':
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            # if not self.training:
            #     print('embed_token_seq_fields:', token_seq_field.shape)            # [bs, L, 一个物品最多含有类别数 ]  [1,|I|,  ]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, max_item_length, seq_len, embed_dim]
            # if not self.training:
            #     print('token_seq_embedding:', token_seq_embedding.shape)        # [bs, L, 一个物品最多含有类别数，d_f]  [1,|I|, ,d_f]
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9
                result = torch.max(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
                result = result.values
            elif self.pooling_mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2, keepdim=True)  # [batch_size, max_item_length, 1, embed_dim]
            # elif self.pooling_mode == 'mean':
            #     masked_token_seq_embedding = token_seq_embedding * mask.float()
            #     result = torch.mean(masked_token_seq_embedding, dim=-2, keepdim=True)
            elif self.pooling_mode == 'gate':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result, _ = self.fusion_layer(masked_token_seq_embedding)  # [B,L,14,D] -> [B,l,1,D]
            elif self.pooling_mode == 'raw':
                return token_seq_embedding, mask
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:                                        # [bs, L, 1, d_f]     [1, |I|, 1, d_f]
            return torch.cat(fields_result, dim=-2)  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    def embed_input_fields(self, user_idx, item_idx, period, select_cate, first_c):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {'user': self.user_feat, 'item': self.item_feat}
        user_item_idx = {'user': user_idx, 'item': item_idx}
        float_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(feature if len(feature.shape) == (2 + (type == 'item')) else feature.unsqueeze(-1))
            if len(float_fields) > 0:
                float_fields = torch.cat(float_fields, dim=-1)  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            # [batch_size, max_item_length, num_float_field]
            # or [batch_size, max_item_length, num_float_field, embed_dim] or None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(token_fields, dim=-1)  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                # if not self.training:
                #     print('===========  user_item_idx[type]: ', user_item_idx[type].shape)  #   [b, L]   [1, |I| ]
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                # if not self.training:
                #     print('==========  feature: ', feature.shape)  # [b,L,一个物品最多同时有几个类]  [bs,L,14]   [1, |I|, 14]
                # print(feature)

                if first_c is not None and first_c == 0:
                    feature[:, :, 0] = 0  # anyway, discard the first category.  Beauty
                if period == 'cd_emb':        # candidate item category. only keep the second category
                    feature[:, :, :select_cate] = 0
                    feature[:, :, select_cate+1:] = 0

                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(token_seq_fields, type)

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat([token_fields_embedding[type], token_seq_fields_embedding[type]],
                                                       dim=-2)
            dense_embedding[type] = float_fields_embedding[type]

        # sparse_embedding[type]
        # shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type]
        # shape: [batch_size, max_item_length, num_float_field]
        #     or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx, first_c=None, period=None, select_cate=None):
        return self.embed_input_fields(user_idx, item_idx, period, select_cate, first_c)


class VanillaAttention2(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim, bias=False), nn.ReLU(True), nn.Linear(attn_dim, 1, bias=False))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, Len, num, H) -> (B, len, 1, H)
        hidden_states = torch.sum(input_tensor * weights.unsqueeze(-1), dim=-2, keepdim=True)
        return hidden_states, weights


class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            'user': list(self.user_feat.interaction.keys()),
            'item': list(self.item_feat.interaction.keys())
        }

        self.types = ['user', 'item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super(FeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)
        self.field_names = {'item': selected_features}

        self.types = ['item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum', 'raw', 'gate']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum', 'gate']!")
        self.get_fields_name_dim()
        self.get_embedding()
        if self.pooling_mode == 'gate':
            self.fusion_layer = VanillaAttention2(self.embedding_size, self.embedding_size)


class CNNLayers(nn.Module):
    r""" CNNLayers

    Args:
        - channels(list): a list contains the channels of each layer in cnn layers
        - kernel(list): a list contains the kernels of each layer in cnn layers
        - strides(list): a list contains the channels of each layer in cnn layers
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                      \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                      \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> m = CNNLayers([1, 32, 32], [2,2], [2,2], 'relu')
        >>> input = torch.randn(128, 1, 64, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 32, 16, 16])
    """

    def __init__(self, channels, kernels, strides, activation='relu', init_method=None):
        super(CNNLayers, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.activation = activation
        self.init_method = init_method
        self.num_of_nets = len(self.channels) - 1

        if len(kernels) != len(strides) or self.num_of_nets != (len(kernels)):
            raise RuntimeError('channels, kernels and strides don\'t match\n')

        cnn_modules = []

        for i in range(self.num_of_nets):
            cnn_modules.append(
                nn.Conv2d(self.channels[i], self.channels[i + 1], self.kernels[i], stride=self.strides[i])
            )
            if self.activation.lower() == 'sigmoid':
                cnn_modules.append(nn.Sigmoid())
            elif self.activation.lower() == 'tanh':
                cnn_modules.append(nn.Tanh())
            elif self.activation.lower() == 'relu':
                cnn_modules.append(nn.ReLU())
            elif self.activation.lower() == 'leakyrelu':
                cnn_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == 'none':
                pass

        self.cnn_layers = nn.Sequential(*cnn_modules)

        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Conv2d):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.cnn_layers(input_feature)


class FMFirstOrderLinear(nn.Module):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """

    def __init__(self, config, dataset, output_dim=1):

        super(FMFirstOrderLinear, self).__init__()
        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        self.LABEL = config['LABEL_FIELD']
        self.device = config['device']
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims, self.token_field_offsets, output_dim)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(np.sum(self.float_field_dims, dtype=np.int32), output_dim)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, output_dim))

        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)

    def embed_float_fields(self, float_fields, embed=True):
        """Calculate the first order score of float feature columns

        Args:
            float_fields (torch.FloatTensor): The input tensor. shape of [batch_size, num_float_field]
            embed (bool): Return the embedding of columns or just the columns itself. Defaults to ``True``.

        Returns:
            torch.FloatTensor: The first order score of float feature columns
        """
        # input Tensor shape : [batch_size, num_float_field]
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, num_float_field, output_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        # [batch_size, 1, output_dim]
        float_embedding = torch.sum(float_embedding, dim=1, keepdim=True)

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Calculate the first order score of token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The first order score of token feature columns
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)
        # [batch_size, 1, output_dim]
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields):
        """Calculate the first order score of token sequence feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]

        Returns:
            torch.FloatTensor: The first order score of token sequence feature columns
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, output_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, output_dim]
            masked_token_seq_embedding = token_seq_embedding * mask.float()
            result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, output_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(torch.cat(fields_result, dim=1), dim=1, keepdim=True)  # [batch_size, 1, output_dim]

    def forward(self, interaction):
        total_fields_embedding = []
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 2:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))

        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None

        # [batch_size, 1, output_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields, embed=True)

        if float_fields_embedding is not None:
            total_fields_embedding.append(float_fields_embedding)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, 1, output_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)
        if token_fields_embedding is not None:
            total_fields_embedding.append(token_fields_embedding)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, 1, output_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)
        if token_seq_fields_embedding is not None:
            total_fields_embedding.append(token_seq_fields_embedding)

        return torch.sum(torch.cat(total_fields_embedding, dim=1), dim=1) + self.bias  # [batch_size, output_dim]


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)
