import pickle
import torch
from tqdm import tqdm
import numpy
from indexing import DenseFlatIndexer
from parameters import RankingParser
from models.E5 import E5Ranker
from models.qwen3 import Qwen3Ranker
from models.llama3 import Llama3Ranker
from transformers import AutoTokenizer

class ECEMetric:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
    
    def __call__(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            num_in_bin = mask.sum().item()
            if num_in_bin > 0:
                accuracy_in_bin = accuracies[mask].float().mean()
                avg_confidence_in_bin = confidences[mask].mean()
                ece += (num_in_bin / len(logits)) * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece.item()

parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()
args = parser.parse_args()
params = args.__dict__

batch_size = 40
device = "cuda:0"
model_type = params.get("found_model", "e5")

# Load model based on type
if model_type == "e5":
    model = E5Ranker(device=device)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model.load_state_dict(torch.load("ms_marco_models/e5/gausian/pytorch_model.bin", weights_only=True))
    query_prefix = "query: "
    doc_prefix = "passage: title: "
elif model_type == "qwen3":
    model = Qwen3Ranker(device=device)
    tokenizer = model.tokenizer
    model.load_state_dict(torch.load("ms_marco_models/qwen3/pytorch_model.bin", weights_only=True))
    query_prefix = ""
    doc_prefix = "title: "
elif model_type == "llama3":
    model = Llama3Ranker(device=device)
    tokenizer = model.tokenizer
    model.load_state_dict(torch.load("ms_marco_models/llama3/pytorch_model.bin", weights_only=True))
    query_prefix = ""
    doc_prefix = "title: "
else:
    raise ValueError(f"Unknown model type: {model_type}")

model.to(device)
model.eval()

documents = pickle.load(open("data/msmarco/eval_documents_100000", "rb"))
eval_queries = pickle.load(open("data/msmarco/eval_queries", "rb"))

# Encode queries
query_encodings = []
questions = [query_prefix + eval_queries[q]["text"] for q in eval_queries.keys()]
relevants = [eval_queries[q]["relevant"][0] for q in eval_queries.keys()]

for i in range(0, len(eval_queries), batch_size):
    curr_batch = tokenizer(questions[i:i+batch_size], max_length=512, padding=True, truncation=True, return_tensors='pt')
    curr_batch = {k: v.to(device) for k, v in curr_batch.items()}
    encs = model.encode_context(curr_batch).tolist()
    query_encodings.extend(encs)

# Encode documents
encodings = []
doc_indices = {}
current_documents = []
index = 0

docs = tqdm(list(documents.keys()))
for key in docs:
    current_documents.append((documents[key][1], documents[key][2]))
    doc_indices[index] = key
    index += 1
    
    if len(current_documents) == batch_size:
        candidates = [doc_prefix + cand[0] + "[SEP] context: " + cand[1] for cand in current_documents]
        batch = tokenizer(candidates, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        encs = model.encode_candidate(batch).tolist()
        encodings.extend(encs)
        current_documents = []

if current_documents:
    candidates = [doc_prefix + cand[0] + "[SEP] context: " + cand[1] for cand in current_documents]
    batch = tokenizer(candidates, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch = {k: v.to(device) for k, v in batch.items()}
    encs = model.encode_candidate(batch).tolist()
    encodings.extend(encs)

# Build index
x_dim, y_dim = len(encodings), len(encodings[0])
vectors = numpy.zeros((x_dim, y_dim), dtype=numpy.float32)
for i in range(len(encodings)):
    vectors[i] = numpy.asarray(encodings[i])

index = DenseFlatIndexer(vector_sz=y_dim)
index.index_data(vectors)
index.index_id_to_db_id = doc_indices

# Search and evaluate
found = index.search(query_encodings, 10)
all_found = sum(1 for i, pred in enumerate(found) if relevants[i] in pred)
all_not_found = len(relevants) - all_found

print(f"found: {all_found} not found: {all_not_found}")
print(f"Recall@10: {all_found / len(relevants):.5f}")

# MRR
all_rr = []
for i, pred in enumerate(found):
    rr = 0
    for rank, entity in enumerate(pred, start=1):
        if entity == relevants[i]:
            rr = 1 / rank
            break
    all_rr.append(rr)
mrr = sum(all_rr) / len(all_rr)
print(f"Mean Reciprocal Rank (MRR): {mrr:.5f}")

# ECE
logits_list, labels_list = [], []
for i in range(len(relevants)):
    query_vec = torch.tensor(query_encodings[i]).to(device)
    candidates = [encodings[index.index_id_to_db_id[j]] for j in found[i]]
    candidate_vecs = torch.tensor(candidates).to(device)
    scores = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), candidate_vecs, dim=1)
    logits_list.append(scores)
    
    relevant_id = relevants[i]
    label = [1 if relevant_id == index.index_id_to_db_id[j] else 0 for j in found[i]]
    labels_list.append(torch.tensor(label).to(device))

logits_tensor = torch.stack(logits_list)
labels_tensor = torch.stack(labels_list).argmax(dim=1)

ece_metric = ECEMetric(n_bins=15)
ece = ece_metric(logits_tensor, labels_tensor)
print(f"Expected Calibration Error (ECE): {ece:.5f}")

try:
    with open("file.txt", "r") as f:
        e = int(f.read())
    with open("val_ece_log.txt", "a+") as f_ece:
        f_ece.write(f"Epoch {e}, Validation ECE = {ece:.4f}\n")
except (FileNotFoundError, ValueError):
    pass

