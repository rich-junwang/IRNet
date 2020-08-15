#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import random
from seq2seq import seq2seq_utils


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            logger.info("Apex LayerNorm is not avaiable. Use the LayerNorm in pytorch.")
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # Position numbers begin at padding_idx+1.
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.max_positions = num_embeddings - self.padding_idx - 1
        self.num_embeddings = num_embeddings

    def forward(self, in_put, offset=0, position_ids=None):
        if position_ids is None:
            positions = seq2seq_utils.make_positions(in_put, self.padding_idx, offset=offset)
        else:
            positions = position_ids + self.padding_idx
        positions = torch.min(positions, (self.num_embeddings - 1) * torch.ones_like(positions))  # compatible to longer sequence
        return super().forward(positions)


class MultiheadAttention(nn.Module):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim  # True for all BART

        assert self.encoder_decoder_attention or qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self, query,
            key_padding_mask=None,  # for (1)
            attn_mask=None,  # for (1.1)
            cached_encoder_key=None,  # [src_len, batch_size, model_dim]
            cached_encoder_value=None,  # [src_len, batch_size, model_dim]
            cached_decoder_key=None,
            cached_decoder_value=None,  # for (3)
    ):
        '''
        different scenarios of multi-head attention:
        (1) encoder side self-attention
        (1.1) decoder side self-attention during training (controlled by attn_mask)
        (2) encoder decoder attention
        (3) decoder side self-attention during inference

        :param query: [tgt_len, batch_size, embed_dim]
        :param key: [src_len, batch_size, embed_dim]
        :param value: [src_len, batch_size, embed_dim]
        :param key_padding_mask: [batch_size, src_len]
        :param incremental_state:
        :param static_kv: previous time steps are cached - no need to recompute key and value if they are static
        :param attn_mask: prevents the attention from looking forward in time
        :return:
        '''
        tgt_len, bsz, embed_dim = query.size()

        # ===== project query/key/value with a linear layer =====
        new_keys = new_values = None
        q = self.q_proj(query) * self.scaling
        if cached_encoder_key is not None and cached_encoder_value is not None:  # encoder decoder attention
            k = cached_encoder_key
            v = cached_encoder_value
        else:  # self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)
            new_keys = k
            new_values = v
            if cached_decoder_key is not None and cached_decoder_value is not None:  # cached previous steps
                k = torch.cat([cached_decoder_key, k], 0)  # 1st dim is for tgt_len
                v = torch.cat([cached_decoder_value, v], 0)

        src_len = k.size(0)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # ==== dot product between query and key =====
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # q [batch_size*num_heads, tgt_len, head_dim]
        # k [batch_size*num_heads, src_len, head_dim]
        # result: [batch_size*num_heads, tgt_len, src_len]
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        # apply attention mask =====
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
            attn_weights += attn_mask

        # apply key_padding mask =====
        if key_padding_mask is not None:  # don't attend to padding symbols
            # import pdb; pdb.set_trace()
            tmp_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            tmp_mask = tmp_mask.expand(-1, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                tmp_mask.to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # softmax over attention weights =====
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float, p=self.dropout, training=self.training,)
        assert v is not None

        # ====== weighted summation =====
        attn_output = torch.bmm(attn_probs, v)
        # attn_probs [batch_size*num_heads, tgt_len, src_len]
        # v [batch_size*num_heads, src_len, head_dim]
        # result: [batch_size*num_heads, tgt_len, head_dim]
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)

        # ====== projection layer =====
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights, new_keys, new_values


# Encoder and Decoder
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MultiheadAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attention_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attention_mask (ByteTensor): binary tensor of shape (seq_len, seq_len)
                attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e8)
            # anything in original attention_mask = 1, becomes -1e8
            # anything in original attention_mask = 0, becomes 0
            # Note that we cannot use -inf here, because at some edge cases,
            # the attention weight (before softmax) for some padded element in query
            # will become -inf, which results in NaN in model parameters
        x, attn_weights, _, _ = self.self_attn(
            x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attention_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MultiheadAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,  # [tgt_len, batch_size, embed_dim]
            decoder_attn_mask=None, decoder_padding_mask=None,
            cached_encoder_padding_mask=None,  # [batch_size, src_len]
            cached_encoder_key=None,  # [src_len, batch_size, model_dim]
            cached_encoder_value=None,  # [src_len, batch_size, model_dim]
            cached_decoder_key=None, cached_decoder_value=None,  # for (3)
    ):
        # ============ decoder side self-attention =====
        residual = x
        x, self_attn_weights, new_decoder_keys, new_decoder_values = self.self_attn(
            x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=decoder_attn_mask,
            cached_decoder_key=cached_decoder_key, cached_decoder_value=cached_decoder_value,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # ============ attention over encoder side =====
        residual = x
        x, encoder_attn_weights, _, _ = self.encoder_attn(
            x,
            key_padding_mask=cached_encoder_padding_mask,
            cached_encoder_key=cached_encoder_key,
            cached_encoder_value=cached_encoder_value)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        # =========== fc layers =====
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, self_attn_weights, new_decoder_keys, new_decoder_values


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx,)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim)

    def forward_embedding(self, input_ids, position_ids=None):
        # embed tokens and positions
        embedded_tokens = self.embed_tokens(input_ids)
        x = embedded_tokens + self.embed_positions(input_ids, position_ids=position_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embedded_tokens

    def forward(self, input_ids, position_ids=None):
        # token embedding + positional embedding
        x, encoder_embedding = self.forward_embedding(input_ids, position_ids=position_ids)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask, 1 means padding positions, 0 means other positions
        encoder_padding_mask = input_ids.eq(self.padding_idx)

        for layer in self.layers:
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = layer(x, encoder_padding_mask)

        return {
            "encoder_out": x,  # T x B x C   Final layer representation
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }


class TransformerDecoder(nn.Module):

    def __init__(self, config, embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = LayerNorm(config.d_model)

        self.new_embed_tokens = None

    def assign_embedding(self, new_embedding):
        self.new_embed_tokens = new_embedding  # [batch_size, vocab_size, model_dim]

    def forward(
            self,
            prev_output_tokens,
            position_offset=0,  # for one_step decoding
            cached_encoder_padding_mask=None,  # [batch_size, src_len]
            cached_encoder_keys=None,  # list of [src_len, batch_size, model_dim]
            cached_encoder_values=None,  # list of [src_len, batch_size, model_dim]
            cached_decoder_keys=None, cached_decoder_values=None,
        ):

        # input positional embedding
        positions = self.embed_positions(prev_output_tokens, offset=position_offset)  # position embedding

        # input token embedding
        if self.new_embed_tokens is None:
            x = self.embed_tokens(prev_output_tokens)
        else:
            x = model_utils.collect_representation(self.new_embed_tokens, prev_output_tokens)

        if positions is not None:
            x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        if position_offset == 0:
            self_attn_mask = seq2seq_utils.build_future_mask(x, time_dim=0)
        else:  # for one_step decoding
            self_attn_mask = None

        # decoder layers
        new_decoder_keys = []  # list of [tgt_len, batch_size, model_dim]
        new_decoder_values = []  # list of [tgt_len, batch_size, model_dim]
        for idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.layerdrop):
                x, layer_self_attn, new_decoder_keys_idx, new_decoder_vaules_idx = layer(
                    x,
                    decoder_attn_mask=self_attn_mask,
                    decoder_padding_mask=self_attn_padding_mask,
                    cached_encoder_padding_mask=cached_encoder_padding_mask,
                    cached_encoder_key=cached_encoder_keys[idx],
                    cached_encoder_value=cached_encoder_values[idx],
                    cached_decoder_key=cached_decoder_keys[idx] if cached_decoder_keys else None,
                    cached_decoder_value=cached_decoder_values[idx] if cached_decoder_values else None,
                )
                new_decoder_keys.append(new_decoder_keys_idx)
                new_decoder_values.append(new_decoder_vaules_idx)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.new_embed_tokens is None:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = torch.bmm(x, self.new_embed_tokens.transpose(1, 2))
        return logits, new_decoder_keys, new_decoder_values


