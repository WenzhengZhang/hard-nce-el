import torch
import torch.nn as nn


NEAR_INF = 1e20


class FullRanker(nn.Module):

    def __init__(self, encoder, device):
        super(FullRanker, self).__init__()
        self.encoder = encoder
        self.dim_hidden = self.encoder.config.hidden_size

        self.score_layer = nn.Sequential(nn.Dropout(0.1),
                                         nn.Linear(self.dim_hidden, 1))
        self.avgCE = nn.CrossEntropyLoss(reduction='mean')
        self.device = device

        # FOLLOWING bert-base-uncased INITIALIZATION
        # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
        self.score_layer[1].weight.data.normal_(
            mean=0.0, std=self.encoder.config.initializer_range)
        self.score_layer[1].bias.data.zero_()

    def forward(self, encoded_pairs, type_marks, input_lens):
        encoded_pairs = encoded_pairs.to(self.device)
        type_marks = type_marks.to(self.device)
        input_lens = input_lens.to(self.device)
        B, C, T = encoded_pairs.size()  # Batch size B, num candidates C, len T
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




