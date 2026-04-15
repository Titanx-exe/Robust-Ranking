import os
import json
import math
import random
import time
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import faiss

### Configuration ###

SEED = 1
MODEL_ID = "meta-llama/Llama-3.2-1B"

# Paths
DATASET_PATH = "blink_training_cleaned_1m.jsonl"
OUTPUT_DIR = "finetuned_models/llama-3.2-1b-lora-dhnm_attn_blink_1m_testrun1"
LOSS_LOG_PATH = "Loss/dhnm_llama_1b_attn_blink_1m_testrun1.jsonl"
EXTERNAL_ENTITIES_PKL = os.environ.get(
    "EXTERNAL_ENTITIES_PKL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ent_descriptions_update.pkl")
)

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NEGATIVES_PER_QUERY = 15
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_SEQ_LENGTH = 1024
TEMPERATURE = 0.05

# DHNM
MINE_EVERY_N_STEPS = 541
PROGRESSIVE_ENTITIES_PER_EPOCH = 495696  # candidate pool grows each epoch

# Checkpointing
RESUME_EPOCH = 0  # set >0 to resume from epoch checkpoint
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_EPOCH = True

# Distributed setup with longer timeout for FAISS operations
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    kwargs_handlers=[process_group_kwargs]
)
DEVICE = accelerator.device


# Utilities

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Grab the final non-padded token's representation."""
    # last real token position for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]


class EntityLinkingDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.pairs = []          # mutable; gets populated with negatives during training
        self.unique_entities: Set[str] = set()
        self.doc_to_gold_entities: Dict[str, Set[str]] = {}

        # Load main training pairs
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                doc_text = obj["document"]
                ent_text = obj["entity"]
                
                self.pairs.append({
                    "document": doc_text,
                    "gold_entity": ent_text,
                    "negatives": []
                })
                self.unique_entities.add(ent_text)


                # Track all valid entities per document to avoid false negatives during mining
                if doc_text not in self.doc_to_gold_entities:
                    self.doc_to_gold_entities[doc_text] = set()
                self.doc_to_gold_entities[doc_text].add(ent_text)


        # Merge external KB entities
        if EXTERNAL_ENTITIES_PKL and os.path.exists(EXTERNAL_ENTITIES_PKL):
            import pickle
            with open(EXTERNAL_ENTITIES_PKL, "rb") as f:
                ext_entities = pickle.load(f)

            # Heuristic: detect if dict is {uri: label} or {label: uri}
            sample_key = next(iter(ext_entities.keys()))
            sample_val = ext_entities[sample_key]
            
            if str(sample_key).startswith(("http", "Q")):
                labels = ext_entities.values()
            elif str(sample_val).startswith(("http", "Q")):
                labels = ext_entities.keys()
            else:
                labels = ext_entities.keys()
            
            for label in labels:
                self.unique_entities.add(str(label))
            
            if accelerator.is_main_process:
                print(f"Loaded external entities from {EXTERNAL_ENTITIES_PKL}")

            
        self.unique_entities_list = sorted(self.unique_entities)
        
        if accelerator.is_main_process:
            print(f"Loaded {len(self.pairs)} document-entity pairs")
            print(f"Total candidate entities: {len(self.unique_entities_list)}")

    def deterministic_shuffle(self, epoch: int) -> None:
        """Shuffle identically on all GPUs using epoch as offset."""
        rng = random.Random(SEED + epoch)
        rng.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]
    

def collate_batch(batch: List[dict], tokenizer, max_length: int) -> Tuple[dict, dict, torch.Tensor]:
    """Build contrastive learning batch with random hard negatives."""
    queries = []
    correct_entities = []
    candidate_pool = set()

    for item in batch:
        queries.append(item["document"])
        gold = item["gold_entity"]
        correct_entities.append(gold)
        
        # Sample negatives
        negs = item["negatives"]
        if len(negs) >= NEGATIVES_PER_QUERY:
            sampled = random.sample(negs, NEGATIVES_PER_QUERY)
        else:
            sampled = negs.copy()
        
        sampled.append(gold)  # ensure positive is in candidates
        candidate_pool.update(sampled)
    
    candidates = list(candidate_pool)
    random.shuffle(candidates)  # shuffle so gold isn't always at end
    

    # Build label indices
    entity_to_idx = {e: i for i, e in enumerate(candidates)}
    labels = [entity_to_idx[e] for e in correct_entities]
    
    # Tokenize separately
    doc_tokens = tokenizer(
        queries, max_length=max_length, padding=True, 
        truncation=True, return_tensors="pt"
    )
    ent_tokens = tokenizer(
        candidates, max_length=max_length, padding=True, 
        truncation=True, return_tensors="pt"
    )


    return doc_tokens, ent_tokens, torch.tensor(labels, dtype=torch.long)


# Model Setup

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="flash_attention_2"
    )
    base_model.config.use_cache = False
    

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    return model, tokenizer


# Hard Negative Mining

def mine_hard_negatives(
    model, 
    tokenizer, 
    dataset: EntityLinkingDataset, 
    epoch: int,
    start_idx: int,
    end_idx: int
):
    """
    Distributed hard negative mining using FAISS.
    Only processes documents from start_idx to end_idx (the upcoming chunk).
    """
    model.eval()
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    is_distributed = world_size > 1
    encode_bs = 64
    
    # Progressive pool: grow candidate set each epoch
    total_ents = len(dataset.unique_entities_list)
    pool_size = (epoch + 1) * PROGRESSIVE_ENTITIES_PER_EPOCH
    
    if pool_size >= total_ents:
        sampled_ents = dataset.unique_entities_list
    else:
        rng = random.Random(SEED + epoch)
        sampled_ents = rng.sample(dataset.unique_entities_list, pool_size)
    
    num_docs = end_idx - start_idx
    
    if rank == 0:
        print(f"\n[Epoch {epoch+1}] Mining {num_docs} docs against {len(sampled_ents)} entities")
    
    # Encode candidate entities (distributed)
    t0 = time.time()
    
    # Split entities across GPUs
    per_gpu = math.ceil(len(sampled_ents) / world_size) if is_distributed else len(sampled_ents)
    my_start = rank * per_gpu if is_distributed else 0
    my_end = min(my_start + per_gpu, len(sampled_ents))
    my_ents = sampled_ents[my_start:my_end]
    
    local_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(my_ents), encode_bs), desc=f"Rank {rank} encoding entities", disable=rank != 0):
            batch = my_ents[i:i+encode_bs]
            tokens = tokenizer(batch, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            out = model(**tokens)
            emb = last_token_pool(out.last_hidden_state, tokens["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            local_embeds.append(emb)
    
    if local_embeds:
        local_tensor = torch.cat(local_embeds, dim=0)
    else:
        local_tensor = torch.empty(0, model.config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    
    # Gather embeddings from all ranks
    if is_distributed:
        local_count = torch.tensor([local_tensor.size(0)], device=DEVICE, dtype=torch.long)
        all_counts = [torch.zeros(1, device=DEVICE, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        counts = [int(c.item()) for c in all_counts]
        max_count = max(counts)
        
        # Pad to max length for all_gather
        padded = torch.zeros(max_count, local_tensor.size(1), device=DEVICE, dtype=local_tensor.dtype)
        padded[:local_tensor.size(0)] = local_tensor
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        
        # Trim padding
        trimmed = [gathered[i][:counts[i]] for i in range(world_size)]
        all_embeds = torch.cat(trimmed, dim=0).float().cpu().numpy()
        del local_tensor, padded, gathered, trimmed
    else:
        all_embeds = local_tensor.float().cpu().numpy()
    
    torch.cuda.empty_cache()
    
    # Build FAISS index (identical on all GPUs)
    index = faiss.IndexFlatIP(all_embeds.shape[1])  # inner product (cosine after norm)
    index.add(all_embeds)
    
    # Encode upcoming documents and search
    docs_per_gpu = math.ceil(num_docs / world_size) if is_distributed else num_docs
    my_doc_start = start_idx + (rank * docs_per_gpu if is_distributed else 0)
    my_doc_end = min(my_doc_start + docs_per_gpu, end_idx)
    
    mined_negatives = {}
    
    with torch.no_grad():
        for i in tqdm(range(my_doc_start, my_doc_end, encode_bs), desc=f"Rank {rank} querying", disable=rank != 0):
            batch_end = min(i + encode_bs, my_doc_end)
            batch_docs = [dataset.pairs[j]["document"] for j in range(i, batch_end)]
            
            tokens = tokenizer(batch_docs, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            out = model(**tokens)
            doc_emb = last_token_pool(out.last_hidden_state, tokens["attention_mask"])
            doc_emb = F.normalize(doc_emb, p=2, dim=-1).float().cpu().numpy()
            
            # Search top-30, filter gold entities
            scores, indices = index.search(doc_emb, 30)
            
            for local_idx, doc_idx in enumerate(range(i, batch_end)):
                doc_text = dataset.pairs[doc_idx]["document"]
                valid_golds = dataset.doc_to_gold_entities[doc_text]
                
                hard_negs = []
                for ent_idx in indices[local_idx]:
                    candidate = sampled_ents[ent_idx]
                    if candidate not in valid_golds:
                        hard_negs.append(candidate)
                
                mined_negatives[doc_idx] = hard_negs
    
    # Synchronize results across ranks
    if is_distributed:
        accelerator.wait_for_everyone()
        all_negs = [None] * world_size
        dist.all_gather_object(all_negs, mined_negatives)
        for neg_dict in all_negs:
            for idx, negs in neg_dict.items():
                dataset.pairs[idx]["negatives"] = negs
    else:
        for idx, negs in mined_negatives.items():
            dataset.pairs[idx]["negatives"] = negs
    
    if rank == 0:
        print(f"[DHNM] Completed in {time.time() - t0:.1f}s")
    
    torch.cuda.empty_cache()


# Training Loop

def train(model, tokenizer, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_SEQ_LENGTH)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    total_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    # Resume from checkpoint if specified
    start_epoch = RESUME_EPOCH
    if start_epoch > 0:
        ckpt_path = os.path.join(OUTPUT_DIR, f"epoch_{start_epoch}")
        if os.path.exists(ckpt_path):
            accelerator.load_state(ckpt_path)
            if accelerator.is_main_process:
                print(f"Resumed from {ckpt_path}")
        else:
            if accelerator.is_main_process:
                print(f"Warning: {ckpt_path} not found, starting from scratch")
            start_epoch = 0

    if accelerator.is_main_process:
        print(f"\nDataset: {len(dataset)} pairs")
        print(f"Effective batch size: {BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS}")
        print(f"Mining frequency: every {MINE_EVERY_N_STEPS} steps\n")
        os.makedirs(os.path.dirname(LOSS_LOG_PATH), exist_ok=True)

    global_step = 0
    docs_per_step = BATCH_SIZE * accelerator.num_processes
    mining_chunk_size = MINE_EVERY_N_STEPS * docs_per_step

    for epoch in range(start_epoch, NUM_EPOCHS):
        dataset.deterministic_shuffle(epoch)
        model.train()
        
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=not accelerator.is_main_process)
        
        for step, (doc_batch, ent_batch, labels) in enumerate(progress):
            # Trigger mining before processing this chunk
            if step % MINE_EVERY_N_STEPS == 0:
                start_idx = step * docs_per_step
                end_idx = min(start_idx + mining_chunk_size, len(dataset.pairs))
                
                unwrapped = accelerator.unwrap_model(model)
                mine_hard_negatives(unwrapped, tokenizer, dataset, epoch, start_idx, end_idx)
                model.train()
            
            with accelerator.accumulate(model):
                # Forward passes
                doc_out = model(**{k: v.to(DEVICE) for k, v in doc_batch.items()})
                ent_out = model(**{k: v.to(DEVICE) for k, v in ent_batch.items()})
                
                # Pool and normalize
                doc_emb = last_token_pool(doc_out.last_hidden_state, doc_batch["attention_mask"])
                ent_emb = last_token_pool(ent_out.last_hidden_state, ent_batch["attention_mask"])
                doc_emb = F.normalize(doc_emb, p=2, dim=-1)
                ent_emb = F.normalize(ent_emb, p=2, dim=-1)
                
                # Contrastive loss
                logits = (doc_emb @ ent_emb.T) / TEMPERATURE
                loss = F.cross_entropy(logits, labels.to(DEVICE))
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            loss_val = accelerator.gather(loss.detach()).mean().item()
            epoch_loss += loss_val
            
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % LOG_EVERY_N_STEPS == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    progress.set_postfix(loss=f"{avg_loss:.4f}")
                    with open(LOSS_LOG_PATH, "a") as f:
                        f.write(json.dumps({"epoch": epoch+1, "step": step, "loss": avg_loss}) + "\n")
        
        # Save epoch checkpoint
        if SAVE_EVERY_EPOCH:
            accelerator.wait_for_everyone()
            epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            accelerator.save_state(epoch_dir)
            
            unwrapped = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped.save_pretrained(epoch_dir)
                tokenizer.save_pretrained(epoch_dir)
                print(f"Saved checkpoint to {epoch_dir}")

    # Final save
    accelerator.wait_for_everyone()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    accelerator.save_state(final_dir)
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training complete. Final model saved to {final_dir}")


def main():
    set_seed(SEED)
    
    dataset = EntityLinkingDataset(DATASET_PATH)
    model, tokenizer = load_model_and_tokenizer()
    
    train(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
