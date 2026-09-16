import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DebertaV3ForSlopDetection(DebertaV2PreTrainedModel):
    def __init__(self, config, num_dropout=5, dropout_rate=0.2, label_smoothing=0.05):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)

        self.num_dropout = num_dropout
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        # Multi-Sample Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropout)])
        # Concatenated Mean + Max pooling doubles hidden_size
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        self.post_init()

    def _convert_to_half(self, module):
        """Helper to cast weights to matching precision"""
        if module.weight is not None and module.weight.dtype != torch.float32:
            pass # Keep precision as it is set by trainer

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Masked Mean & Max Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()

        # Mean Pooling
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Max Pooling
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[input_mask_expanded == 0] = -1e9
        max_pooled = torch.max(sequence_output_masked, 1)[0]

        # Concatenate mean and max pooled representations
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        # Ensure pooled tensor is in same dtype as classifier weights (prevent Float vs Half crash)
        pooled = pooled.to(self.classifier.weight.dtype)

        # Multi-Sample Dropout average
        logits = torch.mean(torch.stack([self.classifier(drop(pooled)) for drop in self.dropouts], dim=0), dim=0)

        loss = None
        if labels is not None:
            # Binary Cross-Entropy with Label Smoothing (0.05)
            # Target 1 becomes 0.95, Target 0 becomes 0.05
            smoothed_labels = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), smoothed_labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
