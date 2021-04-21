import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig
import copy
import torch.nn.functional as F
import math

NEAR_INF = 1e20


class Zeshel(nn.Module):

    def __init__(self, encoder, init_value, device, memb2, inputmark):
        super(Zeshel, self).__init__()
        self.encoder = encoder
        try:
            self.encoder.embeddings.mention_boundary_embeddings
            self.custombert = True
        except AttributeError:
            self.custombert = False

        self.dim_hidden = self.encoder.config.hidden_size

        self.memb2 = memb2
        self.inputmark = inputmark
        if inputmark:
            assert not self.custombert
        if not inputmark:
            if memb2:
                self.mention_marker_embedding = nn.Embedding(2, self.dim_hidden)
            else:
                self.mention_marker_embedding = nn.Embedding(1, self.dim_hidden)

        self.score_layer = nn.Sequential(nn.Dropout(0.1),
                                         nn.Linear(self.dim_hidden, 1))
        self.avgCE = nn.CrossEntropyLoss(reduction='mean')
        self.device = device

        # FOLLOWING bert-base-uncased INITIALIZATION
        # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
        if not inputmark:
            self.mention_marker_embedding.weight.data.normal_(
                mean=0.0, std=self.encoder.config.initializer_range)
        self.score_layer[1].weight.data.normal_(
            mean=0.0, std=self.encoder.config.initializer_range)
        self.score_layer[1].bias.data.zero_()

    def forward(self, encoded_pairs, type_marks, mention_marks, input_lens):
        encoded_pairs = encoded_pairs.to(self.device)
        type_marks = type_marks.to(self.device)
        mention_marks = mention_marks.to(self.device)
        input_lens = input_lens.to(self.device)
        B, C, T = encoded_pairs.size()  # Batch size B, num candidates C, len T

        if self.custombert:
            outputs = self.encoder(encoded_pairs.view(-1, T).long(),
                                   token_type_ids=type_marks.view(-1, T).long(),
                                   mention_ids=mention_marks.view(-1, T).long())
        elif not self.inputmark:
            # Get token embeddings.
            encoded_pairs = encoded_pairs.view(-1, T).long()  # BC x T
            input_embedder = self.encoder.get_input_embeddings()  # V x d
            encoded_pairs = input_embedder(encoded_pairs)  # BC x T x d

            # Get mention mark embeddings.
            if self.memb2:
                mention_marks = mention_marks.view(-1, T).long()
                mention_marks = self.mention_marker_embedding(mention_marks)
            else:
                mention_marks = mention_marks.view(-1, T).float()  # BC x T
                mention_marks = mention_marks.unsqueeze(2)  # BC x T x 1
                mu = self.mention_marker_embedding.weight  # 1 x d
                mention_marks = mention_marks @ mu  # BC x T x d

            # Run encoder on token embeddings + mention mark embeddings.
            outputs = self.encoder(inputs_embeds=encoded_pairs + mention_marks,
                                   token_type_ids=type_marks.view(-1, T).long())
        else:
            # encoded_pairs already contains special mention markers.
            outputs = self.encoder(encoded_pairs.view(-1, T).long(),
                                   token_type_ids=type_marks.view(-1, T).long())

        pooler_output = outputs[1]  # BC x d

        scores = self.score_layer(pooler_output).unsqueeze(1).view(B, C)
        scores.masked_fill_(input_lens == 0, float('-inf'))
        loss = self.avgCE(scores, torch.zeros(B).long().to(self.device))
        max_scores, predictions = scores.max(dim=1)

        return {'loss': loss, 'predictions': predictions,
                'max_scores': max_scores, 'scores': scores}


class PolyAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self, dim=1, attn='basic', residual=False, get_weights=True):
        super().__init__()
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None, values=None):
        """
        Compute attention.
        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)
        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        l1 = torch.matmul(xs, ys.transpose(-1, -2))
        if self.attn == 'sqrt':
            d_k = ys.size(-1)
            l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).unsqueeze(-2)
            l1.masked_fill_(attn_mask, -NEAR_INF)
        l2 = F.softmax(l1, -1, dtype=torch.float).type_as(l1)
        if values is None:
            values = ys
        lhs_emb = torch.matmul(l2, values)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb


class HardAttention(nn.Module):
    """
    Implements simple/classical hard attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs, ys):
        """

        :param xs: (B,T_x,d)
        :param ys: (B,C,T_y,d)
        :return: (B,C)
        """
        bsz, l_x, d = xs.size()
        bsz, C, l_y, d = ys.size()
        scores = (torch.matmul(xs, ys.reshape(bsz, -1, d).transpose(-1,
                                                                    -2)).reshape(
            bsz, l_x, C, l_y).max(-1)[0]).sum(1)
        return scores


class SoftAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self):
        super().__init__()
        self.attention = PolyAttention(dim=2, attn='basic',
                                       get_weights=False)

    def forward(self, xs, ys, values, mask_ys):
        """

        :param xs: (1,C,T_y,d)
        :param ys: (B,T_x,d)
        :param values: (B,T_x,d)
        :param mask_ys: (B,T_x)
        :return: (B,C)
        """
        bsz_x, C, l_y, d = xs.size()
        xs = xs.reshape(bsz_x, -1, d)
        bsz, l_x, d = ys.size()
        attended_embeds = self.attention(xs, ys,
                                         mask_ys=mask_ys,
                                         values=values)  # (B,CT_y,d)
        scores = (attended_embeds * xs).sum(-1).reshape(
            bsz, C, l_y).sum(-1)
        return scores


class UnifiedRetriever(nn.Module):
    def __init__(self, encoder, device, num_codes_mention, num_codes_entity,
                 mention_use_codes, entity_use_codes, attention_type,
                 candidates_embeds=None, evaluate_on=False):
        super(UnifiedRetriever, self).__init__()
        self.mention_use_codes = mention_use_codes
        self.entity_use_codes = entity_use_codes
        self.attention_type = attention_type
        self.mention_encoder = encoder
        self.entity_encoder = copy.deepcopy(encoder)
        self.device = device
        self.loss_fct = CrossEntropyLoss()
        self.num_mention_vecs = num_codes_mention
        self.num_entity_vecs = num_codes_entity
        self.evaluate_on = evaluate_on
        if self.mention_use_codes:
            self.embed_dim = BertConfig().hidden_size
            mention_codes = torch.empty(self.num_mention_vecs, self.embed_dim)
            mention_codes = torch.nn.init.uniform_(mention_codes)
            self.mention_codes = torch.nn.Parameter(mention_codes)
            self.mention_codes_attention = PolyAttention(dim=2, attn='basic',
                                                         get_weights=False)
        if self.entity_use_codes:
            self.embed_dim = BertConfig().hidden_size
            entity_codes = torch.empty(self.num_entity_vecs, self.embed_dim)
            entity_codes = torch.nn.init.uniform_(entity_codes)
            self.entity_codes = torch.nn.Parameter(entity_codes)
            self.entity_codes_attention = PolyAttention(dim=3, attn='basic',
                                                        get_weights=False)
        if self.attention_type == 'soft_attention':
            self.attention = SoftAttention()
        else:
            self.attention = HardAttention()
        self.candidates_embeds = candidates_embeds

    def encode(self, mention_token_ids, mention_masks, candidate_token_ids,
               candidate_masks, entity_token_ids=None, entity_masks=None):
        candidates_embeds = None
        mention_embeds = None
        mention_embeds_masks = None
        if candidate_token_ids is not None:
            candidate_token_ids = candidate_token_ids.to(self.device).long()
            candidate_masks = candidate_masks.to(self.device).long()
            B, C, L = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.view(-1, L)
            candidate_masks = candidate_masks.view(-1, L)
            # B X C X L --> BC X L
            candidates_hiddens = (self.entity_encoder(
                input_ids=candidate_token_ids,
                attention_mask=candidate_masks
            )[0]).reshape(B, C, L, -1)
            candidate_masks = candidate_masks.view(B, C, L)
            if self.entity_use_codes:
                n, d = self.entity_codes.size()
                candidates_embeds = self.entity_codes.unsqueeze(0).unsqueeze(
                    1).expand(B, C, n, d)
                candidates_embeds = self.entity_codes_attention(
                    candidates_embeds, candidates_hiddens,
                    mask_ys=candidate_masks, values=candidates_hiddens)
            else:
                candidates_embeds = candidates_hiddens[:,
                                    :, :self.num_entity_vecs,
                                    :]
        if mention_token_ids is not None:
            mention_token_ids = mention_token_ids.to(self.device).long()
            mention_masks = mention_masks.to(self.device).long()
            mention_hiddens = self.mention_encoder(input_ids=mention_token_ids,
                                                   attention_mask=mention_masks)[
                0]
            B = mention_token_ids.size(0)
            if self.mention_use_codes:
                # m codes m different embeds
                m, d = self.mention_codes.size()
                B, L = mention_token_ids.size()
                mention_codes_embeds = self.mention_codes.unsqueeze(0).expand(B,
                                                                              m,
                                                                              d)
                mention_embeds = self.mention_codes_attention(
                    mention_codes_embeds,
                    mention_hiddens,
                    mask_ys=mention_masks,
                    values=mention_hiddens)
            else:
                mention_embeds = mention_hiddens[:, :self.num_mention_vecs, :]
            mention_embeds_masks = mention_embeds.new_ones(B,
                                                           self.num_mention_vecs).byte()
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_token_ids = entity_token_ids.to(self.device).long()
            entity_masks = entity_masks.to(self.device).long()
            B = entity_token_ids.size(0)
            # B X C X L --> BC X L
            candidates_hiddens = self.entity_encoder(
                input_ids=entity_token_ids,
                attention_mask=entity_masks
            )[0]
            if self.entity_use_codes:
                n, d = self.entity_codes.size()
                candidates_embeds = self.entity_codes.unsqueeze(0).expand(B, n,
                                                                          d)
                candidates_embeds = self.entity_codes_attention(
                    candidates_embeds, candidates_hiddens,
                    mask_ys=candidate_masks, values=candidates_hiddens)
            else:
                candidates_embeds = candidates_hiddens[:, :self.num_entity_vecs,
                                    :]
        return mention_embeds, mention_embeds_masks, candidates_embeds

    def forward(self, mention_token_ids, mention_masks, candidate_token_ids,
                candidate_masks):
        if self.evaluate_on:  # evaluate or get candidates
            mention_embeds, mention_embeds_masks = self.encode(
                mention_token_ids, mention_masks, None, None)[:2]
            bsz, l_x, mention_dim = mention_embeds.size()
            num_cands, l_y, cand_dim = self.candidates_embeds.size()
            if self.attention_type == 'soft_attention':
                scores = self.attention(self.candidates_embeds.unsqueeze(0).to(
                    self.device), mention_embeds, mention_embeds,
                    mention_embeds_masks)
            else:
                scores = (
                    torch.matmul(mention_embeds.reshape(-1, mention_dim),
                                 self.candidates_embeds.reshape(-1,
                                                                cand_dim).t().to(
                                     self.device)).reshape(bsz, l_x,
                                                           num_cands,
                                                           l_y).max(-1)[
                        0]).sum(1)
            return scores
        else:  # train
            B, C, L = candidate_token_ids.size()
            # B x m x d
            #  get  embeds
            mention_embeds, mention_embeds_masks, \
            candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks)
            if self.attention_type == 'soft_attention':
                scores = self.attention(candidates_embeds, mention_embeds,
                                        mention_embeds, mention_embeds_masks)
            else:
                scores = self.attention(mention_embeds, candidates_embeds)
            labels = torch.zeros(B).long().to(self.device)
            loss = self.loss_fct(scores, labels)
            max_scores, predictions = scores.max(dim=1)
            return {'loss': loss, 'predictions': predictions,
                    'max_scores': max_scores, 'scores': scores}


