import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from label_relaxation import lr_torch

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer
#from transformers import AutoTokenizer, AutoModel
#from blinkRanker.model.ranker_base import BertEncoder, get_model_obj


# from biencoderup.common.optimizer import get_bert_optimizer




def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        #ctxt_bert = AutoModel.from_pretrained(params["bert_model"])
        #cand_bert = AutoModel.from_pretrained(params["bert_model"])
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None,device=None,pos_lambda: float = 0.001,
                 neg_lambda: float = 0.01,
                 alpha: float = 0.001,    
                 margin: float = 10):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        if device==None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device=device
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        #self.tokenizer =AutoTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=self.device, weights_only=True)
        else:
            state_dict = torch.load(fname, weights_only=True)
        self.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    # this function is added to compute negative label smoothing
    '''
    def loss_gls(self, logits, labels):
        """
        Computes Generalized Label Smoothing (GLS) loss.
        :param logits: Model logits before softmax, shape [batch_size, num_classes].
        :param labels: True labels, shape [batch_size].
        :return: Smoothed loss.
        """
        batch_size, num_classes = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        smoothing_factor = self.params['label_smoothness']
        # Create smoothed label distribution
        confidence = 1.0 - smoothing_factor
        smoothed_labels = torch.full((batch_size, num_classes), smoothing_factor / (num_classes - 1),
                                     device=self.device)
        smoothed_labels.scatter_(1, labels.unsqueeze(1), confidence)

        

        # Apply Negative Label Smoothing (NLS)
        if smoothing_factor < 0:
            neg_factor = -smoothing_factor
            smoothed_labels = (1 + neg_factor) * smoothed_labels - (neg_factor / (num_classes - 1))
            smoothed_labels = smoothed_labels / smoothed_labels.sum(dim=1, keepdim=True)  # Normalize
        #print("performing negative label_smoothing-------------", smoothed_labels)
        return F.kl_div(log_probs, smoothed_labels, reduction="batchmean")

    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    '''
    '''
    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    '''

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):

        batch_size = cand_vecs.size(0)
        candidate_size = cand_vecs.size(1)
        if not random_negs:
            cand_vecs = cand_vecs.view(batch_size * candidate_size, self.params["max_cand_length"])
        # ret=cand_vecs_view.view(10,5,128)
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        #embedding_cands = embedding_cands.view(batch_size, candidate_size, embedding_cands.size(1))
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_cands = embedding_cands.view(batch_size, candidate_size, embedding_cands.size(1))
            embedding_ctxt = embedding_ctxt.unsqueeze(2)  # batchsize x 1 x embed_size
            # embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_cands, embedding_ctxt)  # batchsize x 1 x 1
            scores = torch.squeeze(scores, dim=2)
            return scores

    def loss_gls_select_nls(self, logits, labels):
        """
        Computes Generalized Label Smoothing (GLS) loss with adaptive smoothing rate.
        """
        print("Applying GLS with adaptive smoothing...")

        # Ensure smoothing_rate is a tensor
        #if not hasattr(self, 'smoothing_rate'):
        #    self.smoothing_rate = torch.full((logits.size(0),), self.params['label_smoothness'], device=logits.device)

        confidence = 1. - self.smoothing_rate  # This is now a tensor

        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        print("nll_loss is --------------------", nll_loss)
        print("smothing rate####################", self.smoothing_rate)
        # Apply per-sample smoothing factor
        loss = confidence * nll_loss + self.smoothing_rate * smooth_loss

        return loss.mean()

    # label_input -- negatives provided

    def selective_nls(self, logits: torch.Tensor, labels: torch.Tensor, min_conf=0.5, max_conf=0.9, max_smooth=-0.1):
        """
        Applies Negative Label Smoothing (NLS) **only when the model is overconfident in the correct label**.

        Args:
            logits (Tensor): Model outputs before softmax, shape [batch_size, num_classes].
            labels (Tensor): Ground truth labels, shape [batch_size].
            min_conf (float): Minimum confidence threshold before applying smoothing.
            max_conf (float): Maximum confidence threshold for full smoothing.
            max_smooth (float): Maximum smoothing factor applied at high confidence.

        Returns:
            Tensor: A tensor of shape [batch_size] containing per-sample smoothing factors.
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected logits to be a Tensor, but got {type(logits)}")

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get probability of the correct class
        correct_probs = probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)

        # Initialize smoothing to zero (no smoothing)
        adaptive_smooth = torch.zeros_like(correct_probs, device=logits.device)

        # **Apply smoothing only to high-confidence samples**
        high_conf_mask = correct_probs < min_conf

        if high_conf_mask.any():  # Apply only if at least one sample is confident
            scaling_factor = (correct_probs[high_conf_mask] - min_conf) / (max_conf - min_conf)
            adaptive_smooth[high_conf_mask] = -0.5

            #adaptive_smooth = torch.clamp(adaptive_smooth, min=0.0, max=max_smooth)  # Ensure valid range
        self.smoothing_rate = adaptive_smooth
        loss = self.loss_gls_select_nls(logits, labels)
        # Logging confidence & smoothing values
        with open('conf_smooth.txt', 'a+') as f1:
            for i in range(len(correct_probs)):
                f1.write(
                    f"Sample {i}: confidence={correct_probs[i].item():.4f}, smoothing={adaptive_smooth[i].item():.4f}\n")

        return loss, logits




    def get_diff(self, scores):
        max_values = scores.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, scores.shape[1])
        diff = max_values - scores
        return diff

    def mbls_forward(self, scores, target):
        alpha = 0.1
        margin = 10
        print("########################## applying mbls ###########################################")
        if scores.dim() > 2:
            scores = scores.view(scores.size(0), scores.size(1), -1)  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)    # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))   # N,H*W,C => N*H*W,C
            scores = scores.view(-1)
            
        loss_ce = F.cross_entropy(scores, target, reduction='mean')
        
        #print("scores is ++++++++++++++++++++++++++++++++++", scores)
        # get logit distance
        diff = self.get_diff(scores)
        #print("differences are ...........................", diff)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + alpha * loss_margin
        #print("loss ce is#################################", loss_ce)
        #print("loss is%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", loss)

        return loss, scores



    def get_reg(self, inputs, targets):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        
        
        neg_dist = max_values.clone().detach() - inputs
        
        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)

        print(f"Reg Pos Part: {(pos.sum() / num_pos).item():.4f}")
        print(f"Reg Neg Part: {(neg.sum() / num_neg).item():.4f}")

        
        return reg


    def acls_forward(self, scores, targets, alpha=0.1):
        if scores.dim() > 2:
            scores = scores.view(scores.size(0), scores.size(1), -1)  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)    # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = F.cross_entropy(scores, targets, reduction="mean")
        #print("????????????????????????????????cross entropy loss is_--------------------------")
        #print(loss_ce)

        loss_reg = self.get_reg(scores, targets)
        loss = loss_ce + self.alpha * loss_reg

        print("#################################### loss_reg is ++++++++++++++++++++++++++++")
        print(loss_reg)
        
        return loss, scores

    #negative label smoothing
    def loss_gls(self, logits, labels):
        # logits: model prediction logits before the soft-max, with size [batch_size, classes]
        # labels: the (noisy) labels for evaluation, with size [batch_size]
        # smooth_rate: could go either positive or negative,
        # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
        #print("labels are...............................", labels)
        #print("labels are...............................", logits)
        print("Applying gls##############################")
        smooth_rate = self.params['label_smoothness']
        confidence = 1. - smooth_rate
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        #print()
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smooth_rate * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return torch.sum(loss) / num_batch

    #original forward function

    def forward(self, context_input, cand_input, label_input=None):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())

        if self.params['adaptive_epoch'] == 'yes':
            if epoch <= self.params['epoch_bound']:
                #continue
                #self.params['label_smoothness'] = self.params['label_smoothness'] - 1.0
                #self.params["learning_rate"] = 3e-5
            #else:
                self.params['label_smoothness'] = 0.0


        smoothing_factor = self.params['label_smoothness']
        #f = open('loss_file.txt', 'a+')
        #f.write("Smoothing factor taken as: " + ' ' + str(smoothing_factor) + '\n')
        #f.close()
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, True)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            print("------------target is ++++++++++++++++++++")
            print(target)
        else:
            target = label_input


        
        #self.smoothing_rate = torch.full((target.size(0),), self.params['label_smoothness'], device=target.device)

        assert target.shape[0] == scores.shape[0], f"Mismatch: target {target.shape}, scores {scores.shape}"

        if self.params['label_relaxation'] == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = lr_torch.LabelRelaxationLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes= num_classes # must match your actual number of classes
            )
            loss = loss_fn(scores, target)
            #print("------------loss is ++++++++++++++++++++")
            #print(loss)

            #print("scores ................................")
            #print(scores)
            return loss, scores


        if self.params['selective_nls'] == 'yes':
            if epoch > self.params['epoch_bound']:
                return self.selective_nls(scores, target)

        if self.params['mbls'] == 'yes':
            return self.mbls_forward(scores, target)

        if self.params['acls'] == 'yes':
            return self.acls_forward(scores, target)


        if smoothing_factor < 0.0:
            loss = self.loss_gls(scores, target)
        else:
            #print("############### target is ################", target)
            #print("+++++++++++++++++ score is +++++++++++++++++", scores)
            #print()
            loss = F.cross_entropy(scores, target, reduction="mean", label_smoothing=smoothing_factor)

        return loss, scores

    def predict(self, context_input, cand_input):
        # flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, False)
        return scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result
