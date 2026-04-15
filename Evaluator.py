import random
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm, trange
import data_processing
import indexing
from models.E5 import E5Ranker
from models.qwen3 import Qwen3Ranker
from models.llama3 import Llama3Ranker
from models.llama_decoder import LlamaDecoderRanker
from models.qwen3_decoder import Qwen3DecoderRanker
from torchmetrics.retrieval import RetrievalMRR

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels

class EvaluatorCrossEncoder:
    def __init__(self,data, params, candidate_size):
        self.entities,self.documents,self.doc_to_ent=data.process_lcquad_file("data/test/lcquad.json")
        self.candidate_size = candidate_size
        self.params = params



    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                canidates=[]
                samples.append((doc,ent))
                while len(canidates)>self.candidate_size:
                    cnd=random.choice(entities.keys())
                    if not cnd in entities:
                        canidates.append(cnd)
        return samples
    def evaluate(self,
        model,random_samples=True
    ):
        if not random_samples:
            samples=[]
        else:
            samples=self.generate_random_samples()
        data_loader=DataLoader(samples, shuffle=True, batch_size=1,
                                  )
        model.eval()
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        for step, batch in enumerate(iter_):
            candidate_input = batch["candidate_encodings"]

            # label_input = batch[0]["label_idx"].to(device)
            # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
            # context_input, candidate_input, label_input = batch
            with torch.no_grad():
                eval_loss, logits = model(candidate_input, label_input=batch["labels"], context_len=32)

                logits = logits.detach().cpu().numpy()
            # Using in-batch negatives, the label ids are diagonal
            label_ids = torch.LongTensor(
                    torch.arange(self.params["eval_batch_size"])
            ).numpy()
            tmp_eval_accuracy, _ = accuracy(logits, batch["labels"].cpu().numpy())

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += candidate_input.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        print("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results

class EvaluatorBiEncoder:
    def __init__(self, params, candidate_size,collator):
        #self.data_loader = data_loader
        self.candidate_size = candidate_size
        self.params = params
        self.collator=collator
        self.entities,self.documents,self.doc_to_ent=data_processing.process_lcquad_file("data/test/lcquad.json")


    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                candidates=[ent]
                while len(candidates)<self.candidate_size:
                    cnd=random.choice(list(self.entities.keys()))
                    if not cnd in entities:
                        candidates.append(cnd)
                samples.append((doc,candidates))
        return samples
    def evaluate(self,model,random_samples=True):
        if not random_samples:
            samples=[]
        else:
            samples=self.generate_random_samples()
        data_loader=DataLoader(samples, shuffle=True, batch_size=100,
                                  collate_fn=self.collator.collate_batch_eval)
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        for step, batch in enumerate(iter_):

            context_input = batch["context_input"]
            candidate_input = batch["candidate_input"]
            labels=[0 for i in range(batch["candidate_input"].size(0))]
            label_ids=torch.tensor(labels,device=model.device)
            #label_input = batch[0]["label_idx"].to(device)
            # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
            # context_input, candidate_input, label_input = batch
            with torch.no_grad():
                eval_loss, logits = model(context_input, candidate_input, label_input=label_ids)

            logits = logits.detach().cpu().numpy()
            # Using in-batch negatives, the label ids are diagonal
            #label_ids = torch.LongTensor(
            #torch.arange(self.params["eval_batch_size"])

            #).numpy()
            tmp_eval_accuracy, _ = accuracy(logits, label_ids.cpu().numpy())

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += context_input.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        print("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results




class IndexEvaluator:

    def __init__(self, params,collator,filehandler,file,tokenizer=None):
        #self.data_loader = data_loader
        #self.candidate_size = candidate_size
        self.params = params
        self.collator=collator
        if tokenizer is not None:
            self.entities, self.documents, self.doc_to_ent = filehandler(file,tokenizer)
        else: self.entities, self.documents, self.doc_to_ent = filehandler(file)
        '''
        if use_lcquad:
            self.entities,self.documents,self.doc_to_ent=data_processing.process_lcquad_file("data/test/lcquad.json")
        else:
            self.entities, self.documents, self.doc_to_ent = data_processing.process_minitaka_file("data/mintaka/mintaka_test.json")
        '''
        self.documents=list(self.documents)
        self.entities=list(self.entities.keys())
    '''
    def __init__(self,params,collator, entities, documents, doc_to_ent):
        self.entities=entities
        self.collator = collator
        self.documents=documents
        self.doc_to_ent=doc_to_ent
        self.params = params
        self.documents = list(self.documents)
        self.entities = list(self.entities.keys())
    '''
    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                candidates=[ent]
                while len(candidates)<self.candidate_size:
                    cnd=random.choice(list(self.entities.keys()))
                    if not cnd in entities:
                        candidates.append(cnd)
                samples.append((doc,candidates))
        return samples
    def evaluate(self,model,random_samples=True,k=10):
        index = indexing.index_entities(model,self.entities, self.collator)

        data_loader=DataLoader(self.documents, shuffle=False, batch_size=50,
                                  collate_fn=self.collator.collate_context)
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Encode Eval Queries")
        doc_encodings=[]
        with torch.no_grad():
            for step, batch in enumerate(iter_):
                if not isinstance(model,E5Ranker) and not isinstance(model, Qwen3Ranker) and not isinstance(model, Llama3Ranker) and not isinstance(model, LlamaDecoderRanker) and not isinstance(model, Qwen3DecoderRanker):

                    context_input = batch
                    #candidate_input = batch["candidate_input"]
                    #labels=[0 for i in range(batch["candidate_input"].size(0))]
                    #label_input = batch[0]["label_idx"].to(device)
                    # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
                    # context_input, candidate_input, label_input = batch
                    encodings=model.encode_context(context_input).tolist()
                else:
                    encodings=model.encode_context(batch)
                doc_encodings.extend(encodings)
        found_ents=index.search(doc_encodings,k)
        mrr = self.evaluate_mrr(found_ents)
        all_labels=[]
        all_scores=[]
        indexes=[]
        all_found=0
        all_not_found=0
        for i in range(len(self.documents)):
            correct_entities=self.doc_to_ent[self.documents[i]]
            prediction=found_ents[i]
            for en in correct_entities:
                if en in prediction:
                    all_found+=1
                else:
                    all_not_found+=1
        print("found:"+str(all_found)+" not found:"+str(all_not_found))
        results=all_found/(all_not_found+all_found)
        #print(result)

        #Adding code for ece computation

        # ---------------------- Compute ECE ----------------------
        ece_metric = ECEMetric(n_bins=15)
        confidences = []
        labels = []

        for i in range(len(self.documents)):
            correct_entities = set(self.doc_to_ent[self.documents[i]])
            prediction = found_ents[i]
            top_pred = prediction[0]  # Top-1 entity

            # If you have similarity scores returned by index.search()
            # Make sure to store them in a variable like: index.last_scores[i]
            # You may need to modify indexing.search() to return scores.
            confidence = index.last_scores[i][0]  # top-1 score
            confidences.append(confidence)

            label = 1 if top_pred in correct_entities else 0
            labels.append(label)

        logits = torch.tensor(confidences).unsqueeze(1)  # shape (N, 1)
        labels = torch.tensor(labels)
        logits = torch.cat([logits, 1 - logits], dim=1)  # fake 2-class logits

        ece_value = ece_metric(logits, labels)
        print(f"**************************** Validation ECE: {ece_value:.5f}")
        with open("epoch.txt", "r") as f:
            data = f.read()  # Reads entire content as a string
            e = int(data)

        with open("val_ece_log.txt", "a+") as f_ece:
            f_ece.write(f"Epoch {e}, Validation ECE = {ece_value:.4f}\n")
        # ---------------------------------------------------------

        return index,results, mrr

    def evaluate_mrr(self, found_ents, random_samples=True, k=10):
        '''
        index = indexing.index_entities(model, self.entities, self.collator)

        data_loader = DataLoader(self.documents, shuffle=False, batch_size=100,
                                 collate_fn=self.collator.collate_context)

        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Encode Eval Queries")

        doc_encodings = []
        for step, batch in enumerate(iter_):
            if not isinstance(model, E5Ranker):
                context_input = batch
                encodings = model.encode_context(context_input).tolist()
            else:
                encodings = model.encode_context(batch)
            doc_encodings.extend(encodings)

        found_ents = index.search(doc_encodings, k)
        '''
        all_rr = []  # List to store reciprocal ranks for MRR calculation

        for i in range(len(self.documents)):
            correct_entities = set(self.doc_to_ent[self.documents[i]])  # Set of correct entities
            prediction = found_ents[i]  # Predicted entities

            # Find the rank of the first correct entity
            reciprocal_rank = 0
            for rank, entity in enumerate(prediction, start=1):
                if entity in correct_entities:
                    reciprocal_rank = 1 / rank
                    break

            all_rr.append(reciprocal_rank)

        mrr = sum(all_rr) / len(all_rr) if all_rr else 0
        print(f"Mean Reciprocal Rank (MRR): {mrr:.5f}")

        return mrr



class ECEMetric:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def __call__(self, logits, labels):
        """
        Computes Expected Calibration Error (ECE).
        Args:
            logits: Tensor of shape (N, C) where C is the number of classes.
            labels: Tensor of shape (N,) with integer class labels.
        Returns:
            Scalar ECE value.
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            num_in_bin = mask.sum().item()

            if num_in_bin > 0:
                accuracy_in_bin = accuracies[mask].float().mean()
                avg_confidence_in_bin = confidences[mask].mean()
                ece += (num_in_bin / logits.size(0)) * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        return ece.item()

