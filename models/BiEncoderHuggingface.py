
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

#from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer, AutoModel
#from blinkRanker.model.ranker_base import BertEncoder, get_model_obj


# from biencoderup.common.optimizer import get_bert_optimizer


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = AutoModel.from_pretrained(params["bert_model"])
        cand_bert = AutoModel.from_pretrained(params["bert_model"])
        #ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        #cand_bert = BertModel.from_pretrained(params['bert_model'])
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
            batch_context,
            batch_candidate

    ):
        embedding_ctxt = None
        if batch_context is not None:
            embedding_ctxt = self.context_encoder(
                batch_context
            )
        embedding_cands = None
        if batch_candidate is not None:
            embedding_cands = self.cand_encoder(
                batch_candidate
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None,device=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
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
        self.tokenizer =AutoTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        #self.tokenizer = BertTokenizer.from_pretrained(
        #    params["bert_model"], do_lower_case=params["lowercase"]
        #)
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

    '''
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
        '''token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )'''
        embedding_context, _ = self.model(
            cands,None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        '''token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )'''
        _, embedding_cands = self.model(
            None, cands
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

        batch_size = cand_vecs.data["input_ids"].size(0)
        candidate_size = cand_vecs.data["input_ids"].size(1)
        if not random_negs:
            cand_vecs = cand_vecs.view(batch_size * candidate_size, self.params["max_cand_length"])
        # ret=cand_vecs_view.view(10,5,128)
        # Encode contexts first
        '''
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        '''
        embedding_ctxt, _ = self.model(
            text_vecs, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        '''
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        '''
        _, embedding_cands = self.model(
            None,cand_vecs
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

        # negative label smoothing
        def loss_gls(self, logits, labels):
            # logits: model prediction logits before the soft-max, with size [batch_size, classes]
            # labels: the (noisy) labels for evaluation, with size [batch_size]
            # smooth_rate: could go either positive or negative,
            # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
            # print("labels are...............................", labels)
            # print("labels are...............................", logits)
            print("Applying gls##############################")
            smooth_rate = self.params['label_smoothness']
            confidence = 1. - smooth_rate
            logprobs = F.log_softmax(logits, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            # print()
            smooth_loss = -logprobs.mean(dim=-1)
            loss = confidence * nll_loss + smooth_rate * smooth_loss
            loss_numpy = loss.data.cpu().numpy()
            num_batch = len(loss_numpy)
            return torch.sum(loss) / num_batch

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None):
        #flag = label_input is None

        smoothing_factor = self.params['label_smoothness']
        flag=True
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            if smoothing_factor < 0.0:
                loss = self.loss_gls(scores, target)
            else:
                print("################ Applying entropy", smoothing_factor)
                loss = F.cross_entropy(scores, target, reduction="mean", label_smoothing=smoothing_factor)
        else:
            # loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            if smoothing_factor < 0.0:
                loss = self.loss_gls(scores, label_input)
            else:
                print("################ Applying entropy", smoothing_factor)
                loss = F.cross_entropy(scores, label_input, reduction="mean", label_smoothing=smoothing_factor)

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

    def forward(self, batch):
        output_bert = self.bert_model(
            **batch
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_bert.pooler_output
        else:
            #embeddings = output_bert[:, 0, :]
            embeddings = output_bert.last_hidden_state[:,0,:]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result
