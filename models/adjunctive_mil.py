"""
The defnition of Adjunctive MIL is DIFFERENT in this implementation!
I'm trying the "same classifier" approach at the moment. Meaning the classifier is shared
between positive and negative evidence but the attention is calculate differently for each.
"""

from typing import Sequence

import torch
import torch.nn as nn

from xai.explanation import xMIL


def get_adjunctive_mil_model(input_dim=1024, num_classes=2, hidden_dim=256, device='cpu'):
    model = DefaultMILGraph(
        pointer=DefaultAttentionModule(
            hidden_activation=nn.LeakyReLU(0.2), 
            hidden_dims=[hidden_dim, hidden_dim], 
            input_dims=input_dim,
            # now the pointer outputs two scores per patch (positive and negative)
            output_dim=2
        ),
        classifier=AdjunctiveClassifier(
            hidden_dims=[hidden_dim, hidden_dim], 
            input_dims=input_dim, 
            output_dims=num_classes
        ),
        num_classes=num_classes,
        device=device
    )
    return model


class DefaultMILGraph(torch.nn.Module):
    def __init__(
        self,
        classifier: torch.nn.Module,
        pointer: torch.nn.Module,
        num_classes: int,
        device: str,
    ):
        super().__init__()
        self.classifier = classifier
        self.pointer = pointer
        self.n_classes = num_classes
        self.patch_logits = None
        self.pos_patch_logits = None
        self.neg_patch_logits = None
        self.attention_scores = None
        self.device = device

    def forward_fn(self, features, bag_sizes):
        return self.forward(features, bag_sizes)

    def forward(self, features, bag_sizes):
        """
        :param features: (#patches, input_dim)
        :param bag_sizes: (list or tensor of bag sizes)
        :return: (#bags, num_classes)
        """
        attention_scores = self.pointer(features, bag_sizes)
        cls_out_dict = self.classifier(features, attention_scores, bag_sizes)
        self.attention_scores = attention_scores
        self.patch_logits = cls_out_dict.get('patch_logits')
        self.pos_patch_logits = cls_out_dict.get('pos_patch_logits')
        self.neg_patch_logits = cls_out_dict.get('neg_patch_logits')
        return cls_out_dict['logits']


class StableSoftmax(torch.nn.Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()


class DefaultAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = nn.ReLU(),
        # now we add an output_dim parameter; for our multi-head setting, set this to 2.
        output_dim: int = 2,
        output_activation: torch.nn.Module = StableSoftmax(dim=0),
        use_batch_norm: bool = True,
        track_bn_stats: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats
        self.model = self.build_model()

    def build_model(self):
        # modify the last layer to output "output_dim" instead of 1.
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dim]
        layers = []
        for i, (nodes_in, nodes_out) in enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:])):
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(nodes_out, track_running_stats=self.track_bn_stats))
                layers.append(self.hidden_activation)
        model = nn.Sequential(*layers)
        return model

    def bag_activation(self, scores, bag_sizes):
        # scores: (total_patches, output_dim) where output_dim == 2
        scores_softmax = []
        start = 0
        for bag_size in bag_sizes:
            bag_scores = scores[start:start + bag_size]  # shape: (bag_size, 2)
            # Apply softmax independently on each head (i.e. for each column) over the bag's patches.
            pos_attention = torch.softmax(bag_scores[:, 0], dim=0).unsqueeze(-1)
            neg_attention = torch.softmax(bag_scores[:, 1], dim=0).unsqueeze(-1)
            # Concatenate back to shape (bag_size, 2)
            bag_attention = torch.cat([pos_attention, neg_attention], dim=-1)
            scores_softmax.append(bag_attention)
            start += bag_size
        return torch.cat(scores_softmax, dim=0)

    def forward(self, features, bag_sizes):
        out = self.model(features)  # shape: (total_patches, output_dim=2)
        attention = self.bag_activation(out, bag_sizes)
        return attention


class AdjunctiveClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        layers = []
        for i, (nodes_in, nodes_out) in enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:])):
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
        model = torch.nn.Sequential(*layers)
        return model

    @staticmethod
    def aggregate_patch_scores(patch_scores, bag_sizes):
        res = []
        start = 0
        for bag_size in bag_sizes:
            patches_probs = patch_scores[start:start + bag_size]  # (bag_size, n_class)
            bag_probs = patches_probs.sum(dim=0, keepdims=True)    # (1, n_class)
            res.append(bag_probs)
            start += bag_size
        return torch.cat(res, dim=0)

    # def forward(self, features, attention, bag_sizes):
    #     # classifier outputs patch logits of shape: (total_patches, n_classes)
    #     patch_logits = self.model(features)
    #     # attention has shape: (total_patches, 2) with first column = positive, second = negative
    #     pos_attention = attention[:, 0:1]  # shape: (total_patches, 1)
    #     neg_attention = attention[:, 1:2]  # shape: (total_patches, 1)
    #     # Compute net attention as difference between positive and negative signals.
    #     net_attention = pos_attention - neg_attention  # shape: (total_patches, 1)
    #     # Multiply the patch logits elementwise by the net attention.
    #     weighted_patch_logits = net_attention * patch_logits
    #     pos_evidence = pos_attention * patch_logits
    #     neg_evidence = neg_attention * patch_logits
    #     bag_logits = self.aggregate_patch_scores(weighted_patch_logits, bag_sizes)
    #     classifier_out_dict = {
    #         'logits': bag_logits,
    #         'pos_patch_logits': pos_evidence,
    #         'neg_patch_logits': neg_evidence,
    #         'patch_logits': weighted_patch_logits,
    #     }
    #     return classifier_out_dict
    
    # VERSION WITH RELU ON POS AND NEG LOGITS
    def forward(self, features, attention, bag_sizes):
        # raw logits from the shared classifier (can be negative or positive)
        raw_logits = self.model(features)  # shape: (total_patches, n_classes)
        
        # Split attention into positive and negative parts.
        pos_attention = attention[:, 0:1]  # shape: (total_patches, 1)
        neg_attention = attention[:, 1:2]  # shape: (total_patches, 1)
        
        # Force positive branch to only produce nonnegative values:
        pos_logits = torch.relu(raw_logits)         # if raw_logits < 0 => 0; if > 0, passes through
        # Force negative branch to only produce nonnegative values:
        neg_logits = torch.relu(-raw_logits)          # if raw_logits > 0 => 0; if < 0, becomes -raw_logits (> 0)
        
        # Compute evidence per branch:
        pos_evidence = pos_attention * pos_logits
        neg_evidence = neg_attention * neg_logits
        
        # The net evidence (which can be positive or negative) is then:
        net_evidence = pos_evidence - neg_evidence

        # Aggregate patch-level net evidence to form bag-level logits:
        bag_logits = self.aggregate_patch_scores(net_evidence, bag_sizes)
        
        classifier_out_dict = {
            'logits': bag_logits,
            'pos_patch_logits': pos_evidence,
            'neg_patch_logits': neg_evidence,
            'patch_logits': net_evidence,
        }
        return classifier_out_dict


class xAdjunctiveMIL(xMIL):
    def __init__(self, model, explained_class=None):
        super().__init__()
        self.model = model
        self.device = model.device
        self.explained_class = explained_class

    def attention_map(self, batch):
        """
        Returns the attention scores for the patches when the model is applied on the input data.
        The returned attention has two columns per patch (for positive and negative evidence).
        """
        self.model.eval()
        features, bag_sizes = batch['features'].to(self.device), batch['bag_size'].to(self.device)
        self.model.forward(features, bag_sizes)
        # self.model.attention_scores has shape (total_patches, 2)
        attn = self.model.attention_scores.detach().cpu().numpy()
        pos_attn, neg_attn = attn[:, 0:1].squeeze(), attn[:, 1:2].squeeze()
        net_attn = pos_attn - neg_attn
        return pos_attn, neg_attn, net_attn

    def explain_patch_scores(self, batch):
        """
        Returns a dictionary with keys 'pos_patch_logits', 'neg_patch_logits', and 'patch_logits'
        for the explained class.
        """
        self.model.eval()
        features, bag_sizes = batch['features'].to(self.device), batch['bag_size'].to(self.device)
        self.model.forward(features, bag_sizes)
        explained_class = self.set_explained_class(self.explained_class)
        return self.model.pos_patch_logits[:, explained_class].detach().cpu().numpy().squeeze(), self.model.neg_patch_logits[:, explained_class].detach().cpu().numpy().squeeze(), self.model.patch_logits[:, explained_class].detach().cpu().numpy().squeeze()
        
