"""
Fix missing entity descriptions in pkl files.

This script:
1. Collects ALL entities from ALL NIF datasets (AIDA, ACE2004, AQUAINT, etc.)
2. Finds which ones are missing from entity_descriptions.pkl
3. Fetches their descriptions from SPARQL in batches of 50
4. Merges them into entity_descriptions.pkl
5. Re-runs the update_labels.py logic to regenerate ent_descriptions_update.pkl
"""

import pickle
import os
import sys
import time
from tqdm import tqdm
from transformers import AutoTokenizer

project_root = os.environ.get("ROBUST_RANKING_ROOT", os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_processing import Aida_joint_el, get_text_knowledge_for_entities

# ============================================================
# STEP 1: Collect ALL entities from ALL NIF datasets
# ============================================================
print("=" * 80)
print("STEP 1: Collecting entities from ALL NIF datasets")
print("=" * 80)

base_path = os.path.join(project_root, "data/aida/wikidata")

nif_files = [
    # (dataset_name, train_path, test_path)
    ("ACE2004",       f"{base_path}/ace2004_splits/ACE2004_train",           f"{base_path}/ace2004_splits/ACE2004_testa"),
    ("AQUAINT",       f"{base_path}/AQUAINT_splits/AQUAINT_train",           f"{base_path}/AQUAINT_splits/AQUAINT_testa"),
    ("AIDA",          f"{base_path}/aida_splits/aida_train",                 f"{base_path}/aida_splits/aida_testa"),
    ("iitb-fix",      f"{base_path}/iitb-fix_splits/iitb-fix_train",         f"{base_path}/iitb-fix_splits/iitb-fix_testa"),
    ("KORE50",        f"{base_path}/KORE50_splits/KORE50_train",             f"{base_path}/KORE50_splits/KORE50_testa"),
    ("MSNBC",         f"{base_path}/MSNBC_splits/MSNBC_train",               f"{base_path}/MSNBC_splits/MSNBC_testa"),
    ("N3-Reuters-128",f"{base_path}/N3-Reuters-128_splits/N3-Reuters-128_train", f"{base_path}/N3-Reuters-128_splits/N3-Reuters-128_testa"),
    ("N3-RSS-500",    f"{base_path}/N3-RSS-500_splits/N3-RSS-500_train",     f"{base_path}/N3-RSS-500_splits/N3-RSS-500_testa"),
    ("spotlight",     f"{base_path}/spotlight_splits/spotlight_train",         f"{base_path}/spotlight_splits/spotlight_testa"),
]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
dp = Aida_joint_el()

all_nif_entities = set()
for name, train_path, test_path in nif_files:
    for split_name, path in [("train", train_path), ("test-a", test_path)]:
        if os.path.exists(path):
            try:
                ents, _, _ = dp.read_ds_to_list(path, tokenizer)
                all_nif_entities.update(ents.keys())
                print(f"  ✓ {name} {split_name}: {len(ents)} entities")
            except Exception as e:
                print(f"  ✗ {name} {split_name}: ERROR - {e}")
        else:
            print(f"  ✗ {name} {split_name}: FILE NOT FOUND")

print(f"\nTotal unique NIF entities: {len(all_nif_entities)}")

# ============================================================
# STEP 2: Find which entities are MISSING from the pkl
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Finding missing entities")
print("=" * 80)

pkl_path = os.path.join(project_root, "data/entity_descriptions.pkl")
pkl_update_path = os.path.join(project_root, "data/ent_descriptions_update.pkl")

existing_descs = pickle.load(open(pkl_path, "rb"))
print(f"Existing pkl has: {len(existing_descs)} entities")

missing_entities = [e for e in all_nif_entities if e not in existing_descs]
already_have = [e for e in all_nif_entities if e in existing_descs]

print(f"NIF entities already in pkl: {len(already_have)}")
print(f"NIF entities MISSING from pkl: {len(missing_entities)}")

if len(missing_entities) == 0:
    print("\nNo missing entities! Nothing to do.")
    sys.exit(0)

# ============================================================
# STEP 3: Fetch descriptions from SPARQL in batches of 50
# ============================================================
print("\n" + "=" * 80)
print(f"STEP 3: Fetching {len(missing_entities)} entities from SPARQL (batches of 50)")
print("=" * 80)

new_descriptions = {}
failed_entities = []
batch_size = 50
batches = [missing_entities[i:i+batch_size] for i in range(0, len(missing_entities), batch_size)]

for i, batch in enumerate(tqdm(batches, desc="SPARQL batches")):
    try:
        result = get_text_knowledge_for_entities(batch)
        new_descriptions.update(result)
        
        # Track entities that SPARQL returned nothing for
        for ent in batch:
            if ent not in result:
                failed_entities.append(ent)
        
        # Small delay to be nice to the SPARQL endpoint
        time.sleep(0.5)
        
    except Exception as e:
        print(f"\n  ✗ Batch {i+1} failed: {e}")
        failed_entities.extend(batch)
        time.sleep(2)  # Longer delay after failure

print(f"\nSPARQL results:")
print(f"  Successfully fetched: {len(new_descriptions)} entities")
print(f"  Failed/not found:     {len(failed_entities)} entities")

if len(failed_entities) > 0:
    print(f"\n  Entities with no SPARQL results:")
    for ent in failed_entities[:20]:
        print(f"    {ent}")
    if len(failed_entities) > 20:
        print(f"    ... and {len(failed_entities) - 20} more")

# ============================================================
# STEP 4: Merge into entity_descriptions.pkl
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: Merging into entity_descriptions.pkl")
print("=" * 80)

# Backup the original
backup_path = pkl_path + ".backup"
if not os.path.exists(backup_path):
    pickle.dump(existing_descs, open(backup_path, "wb"))
    print(f"  ✓ Backup saved to {backup_path}")
else:
    print(f"  ℹ Backup already exists at {backup_path}")

# Merge
existing_descs.update(new_descriptions)
pickle.dump(existing_descs, open(pkl_path, "wb"))
print(f"  ✓ Updated entity_descriptions.pkl: {len(existing_descs)} entities (was {len(existing_descs) - len(new_descriptions)})")

# ============================================================
# STEP 5: Regenerate ent_descriptions_update.pkl
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: Regenerating ent_descriptions_update.pkl (update_labels.py logic)")
print("=" * 80)

# This is the exact logic from update_labels.py
all_labels = {}
for k in existing_descs.keys():
    lb = existing_descs[k].replace("label:", "title")
    lb = lb.replace("alt", "")
    while "[SEP]" in lb:
        lb = lb.replace("[SEP]", "")
    all_labels[k] = lb

# Backup
backup_update_path = pkl_update_path + ".backup"
if not os.path.exists(backup_update_path):
    pickle.dump(pickle.load(open(pkl_update_path, "rb")), open(backup_update_path, "wb"))
    print(f"  ✓ Backup saved to {backup_update_path}")
else:
    print(f"  ℹ Backup already exists at {backup_update_path}")

pickle.dump(all_labels, open(pkl_update_path, "wb"))
print(f"  ✓ Updated ent_descriptions_update.pkl: {len(all_labels)} entities")

# ============================================================
# STEP 6: Verification
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: Verification")
print("=" * 80)

# Reload and verify
verify_desc = pickle.load(open(pkl_path, "rb"))
verify_update = pickle.load(open(pkl_update_path, "rb"))

print(f"entity_descriptions.pkl:     {len(verify_desc)} entities")
print(f"ent_descriptions_update.pkl: {len(verify_update)} entities")

# Check NIF coverage now
nif_found = sum(1 for e in all_nif_entities if e in verify_desc)
print(f"\nNIF entity coverage: {nif_found}/{len(all_nif_entities)} ({100*nif_found/len(all_nif_entities):.1f}%)")
print(f"Previously was: {len(already_have)}/{len(all_nif_entities)} ({100*len(already_have)/len(all_nif_entities):.1f}%)")

# Show some resolved examples
print("\nSample resolved entities (previously missing):")
for ent in list(new_descriptions.keys())[:5]:
    print(f"  ✓ {ent}")
    print(f"    → {verify_update.get(ent, 'NOT FOUND')[:100]}")

print("\n" + "=" * 80)
print("DONE! Both pkl files have been updated.")
print("=" * 80)
