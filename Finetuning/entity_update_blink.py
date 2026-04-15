"""
entity_update_blink.py

This script processes the BLINK dataset to identify all unique Wikipedia entities,
maps them to Wikidata URIs via the Wikipedia API, and fetches any missing
entity descriptions using SPARQL. It then updates the entity description
pickle files with the newly fetched data.

Features:
- Extracts Wikipedia entity IDs from the BLINK dataset efficiently (streamed reading).
- Maps Wikipedia IDs to Wikidata URIs in batches.
- Identifies entities not yet present in `entity_descriptions.pkl`.
- Fetches missing descriptions and updates `entity_descriptions.pkl` and `ent_descriptions_update.pkl`.
- Displays continuous progress bars with estimated time of completion and entity counts using `tqdm`.
"""

import json
import os
import sys
import pickle
import time
import requests
import logging
from tqdm import tqdm

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOG_FILE = os.path.join(SCRIPT_DIR, "entity_update.log")

# Ensure project root is in PYTHONPATH to import data_processing
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from data_processing import get_text_knowledge_for_entities
except ImportError:
    logger.error("Could not import 'get_text_knowledge_for_entities' from data_processing.")
    logger.error(f"Make sure 'data_processing.py' exists in the project root directory: {PROJECT_ROOT}")
    sys.exit(1)

# --- Configuration ---
DATASET_PATH = os.environ.get(
    "BLINK_DATASET_PATH",
    os.path.join(PROJECT_ROOT, "data", "aida", "wikidata", "BLINK", "blink-train-kilt.jsonl")
)
ENTITY_DESC_PKL = os.path.join(PROJECT_ROOT, "data", "entity_descriptions.pkl")
ENTITY_DESC_UPDATE_PKL = os.path.join(PROJECT_ROOT, "data", "ent_descriptions_update.pkl")


def load_jsonl_entities(file_path: str) -> dict:
    """
    Streamingly load a .jsonl file and collect unique Wikipedia entities.
    We iterate line by line to dramatically save memory compared to loading the whole dataset.
    
    Returns:
        dict: Mapping of wikipedia_id (str) -> wikipedia_title (str)
    """
    wp_entities = {}
    logger.info(f"Reading dataset to extract Wikipedia entities: {file_path}")
    
    # First, count total lines to give an accurate progress bar with ETA
    logger.info("Counting total records in dataset for ETA estimation...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        
    logger.info(f"Total records found: {total_lines:,}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        # Use mininterval to avoid spamming log files with too many progress updates
        for line in tqdm(f, total=total_lines, desc="Parsing BLINK records", unit="record", mininterval=5.0):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                for out in record.get("output", []):
                    for prov in out.get("provenance", []):
                        wp_id = str(prov.get("wikipedia_id", ""))
                        title = prov.get("title", out.get("answer", ""))
                        if wp_id:
                            wp_entities[wp_id] = title
            except json.JSONDecodeError:
                # Silently skip malformed lines during parsing
                continue
                
    return wp_entities


def map_wikipedia_ids_to_wikidata(wp_ids: list[str], batch_size: int = 50) -> dict:
    """
    Map Wikipedia page IDs to Wikidata URIs via the Wikipedia API.
    
    Uses a persistent session with a proper User-Agent header, exponential
    backoff on failures (429/5xx), and adaptive delay between requests.
    
    Returns:
        dict: Mapping of wikipedia_id (str) -> wikidata_uri (str)
    """
    API_URL = "https://en.wikipedia.org/w/api.php"
    MAX_RETRIES = 5          # Max retries per batch before giving up
    BASE_DELAY = 0.5         # Seconds between successful requests
    BACKOFF_FACTOR = 2.0     # Multiplier for exponential backoff on failure
    LOG_EVERY_N = 500        # Log progress every N batches
    
    wp_to_wd = {}
    permanently_failed = []
    
    # Use a persistent session for connection reuse and consistent headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "RobustRankingEntityUpdate/1.0 (https://dice-research.org; research script) python-requests"
    })
    
    batches = [wp_ids[i:i+batch_size] for i in range(0, len(wp_ids), batch_size)]
    total_batches = len(batches)
    
    logger.info(f"Mapping {len(wp_ids):,} Wikipedia IDs to Wikidata URIs ({total_batches} batches)...")
    logger.info(f"Estimated time: ~{(total_batches * BASE_DELAY) / 3600:.1f} hours at {BASE_DELAY}s/batch")
    
    start_time = time.time()
    
    for batch_num, batch in enumerate(batches):
        ids_str = "|".join(batch)
        
        params = {
            "action": "query",
            "pageids": ids_str,
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "format": "json",
        }
        
        # Retry loop with exponential backoff
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(API_URL, params=params, timeout=30)
                
                # Handle rate limiting (429) and server errors (5xx) with backoff
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait_time = BASE_DELAY * (BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        f"Batch {batch_num+1}/{total_batches}: HTTP {resp.status_code}. "
                        f"Attempt {attempt+1}/{MAX_RETRIES}. Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    wd_qid = page_data.get("pageprops", {}).get("wikibase_item")
                    if wd_qid:
                        wd_uri = f"http://www.wikidata.org/entity/{wd_qid}"
                        wp_to_wd[page_id] = wd_uri
                
                success = True
                break  # Success — exit retry loop
                
            except requests.exceptions.RequestException as e:
                wait_time = BASE_DELAY * (BACKOFF_FACTOR ** attempt)
                logger.warning(
                    f"Batch {batch_num+1}/{total_batches}: Request error: {e}. "
                    f"Attempt {attempt+1}/{MAX_RETRIES}. Waiting {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
        
        if not success:
            permanently_failed.extend(batch)
            logger.error(f"Batch {batch_num+1}/{total_batches}: Failed after {MAX_RETRIES} retries. Skipping {len(batch)} IDs.")
        
        # Log progress every N batches
        if (batch_num + 1) % LOG_EVERY_N == 0 or (batch_num + 1) == total_batches:
            elapsed = time.time() - start_time
            rate = (batch_num + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_batches - batch_num - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {batch_num+1}/{total_batches} batches "
                f"({100*(batch_num+1)/total_batches:.1f}%) | "
                f"Mapped: {len(wp_to_wd):,} IDs | "
                f"Rate: {rate:.1f} batch/s | "
                f"ETA: {remaining/60:.0f} min"
            )
        
        # Polite delay between requests
        time.sleep(BASE_DELAY)
            
    logger.info(f"Successfully mapped {len(wp_to_wd):,} Wikipedia IDs to Wikidata URIs.")
    if permanently_failed:
        logger.info(f"Permanently failed (after all retries): {len(permanently_failed):,} IDs.")
        
    return wp_to_wd


def main():
    logger.info("="*50)
    logger.info("   Starting BLINK Entity Pickle Update Script")
    logger.info("="*50)
    
    # 1. Extract Wikipedia entities from BLINK data
    wp_entities = load_jsonl_entities(DATASET_PATH)
    logger.info(f"Unique Wikipedia entities found in BLINK: {len(wp_entities):,}")

    # 2. Map Wikipedia IDs back to Wikidata URIs
    wp_id_list = list(wp_entities.keys())
    wp_to_wd = map_wikipedia_ids_to_wikidata(wp_id_list)
    
    # 3. Load existing pickle file
    logger.info(f"Loading existing entity descriptions from: {ENTITY_DESC_PKL}")
    if os.path.exists(ENTITY_DESC_PKL):
        try:
            entity_text_dict = pickle.load(open(ENTITY_DESC_PKL, "rb"))
            logger.info(f"Loaded existing descriptions. Total entities: {len(entity_text_dict):,}")
        except Exception as e:
            logger.error(f"Error loading pickle: {e}")
            sys.exit(1)
    else:
        logger.warning("Pickle file not found. Creating a new empty dictionary.")
        entity_text_dict = {}

    # 4. Find which mapped Wikidata URIs are currently missing from the local pickle
    all_wd_uris = list(wp_to_wd.values())
    missing_uris = [uri for uri in all_wd_uris if uri not in entity_text_dict]
    already_have = [uri for uri in all_wd_uris if uri in entity_text_dict]
    
    logger.info(f"BLINK Wikidata URIs already in local pickle: {len(already_have):,}")
    logger.info(f"BLINK Wikidata URIs MISSING and to be fetched: {len(missing_uris):,}")
    
    # 5. Connect and fetch missing entity descriptions using SPARQL
    new_descriptions = {}
    if len(missing_uris) > 0:
        logger.info(f"Fetching descriptions for {len(missing_uris):,} missing entities...")
        failed_entities = []
        batch_size = 50
        batches = [missing_uris[i:i+batch_size] for i in range(0, len(missing_uris), batch_size)]
        
        for batch in tqdm(batches, desc="Fetching via SPARQL", unit="batch", mininterval=5.0):
            try:
                result = get_text_knowledge_for_entities(batch)
                new_descriptions.update(result)
                for ent in batch:
                    if ent not in result:
                        failed_entities.append(ent)
                time.sleep(0.5)
            except Exception as e:
                failed_entities.extend(batch)
                logger.warning(f"Batch fetch failed: {e}. Retrying after sleep.")
                time.sleep(2)
                
        logger.info(f"Successfully fetched: {len(new_descriptions):,} new entity descriptions.")
        logger.info(f"Failed to fetch or not found: {len(failed_entities):,} entities.")
    else:
        logger.info("All BLINK entities already have descriptions! No SPARQL fetching needed.")
        
    # 6. Update Local Pickles if new description was loaded
    if len(new_descriptions) > 0:
        logger.info("Updating local pickle files...")
        
        # Create a Backup for ENTITY_DESC_PKL
        backup_path = ENTITY_DESC_PKL + ".backup"
        if not os.path.exists(backup_path):
            pickle.dump(entity_text_dict, open(backup_path, "wb"))
            logger.info(f"Saved backup of original entity_descriptions.pkl to: {backup_path}")
            
        # Merge descriptions
        old_count = len(entity_text_dict)
        entity_text_dict.update(new_descriptions)
        
        # Save updated ENTITY_DESC_PKL
        pickle.dump(entity_text_dict, open(ENTITY_DESC_PKL, "wb"))
        logger.info(f"Updated {ENTITY_DESC_PKL}: {old_count:,} -> {len(entity_text_dict):,} entities.")
        
        # Re-generate ENTITY_DESC_UPDATE_PKL with cleaned/modified descriptions
        logger.info("Regenerating updated labels pickle (ent_descriptions_update.pkl)...")
        all_labels = {}
        for k in tqdm(entity_text_dict.keys(), desc="Processing labels", unit="entity", mininterval=5.0):
            lb = entity_text_dict[k].replace("label:", "title")
            lb = lb.replace("alt", "")
            while "[SEP]" in lb:
                lb = lb.replace("[SEP]", "")
            all_labels[k] = lb
            
        backup_update_path = ENTITY_DESC_UPDATE_PKL + ".backup"
        if os.path.exists(ENTITY_DESC_UPDATE_PKL) and not os.path.exists(backup_update_path):
            # Backup first 
            pickle.dump(pickle.load(open(ENTITY_DESC_UPDATE_PKL, "rb")), open(backup_update_path, "wb"))
            logger.info(f"Saved backup of original ent_descriptions_update.pkl to: {backup_update_path}")
            
        pickle.dump(all_labels, open(ENTITY_DESC_UPDATE_PKL, "wb"))
        logger.info(f"Updated {ENTITY_DESC_UPDATE_PKL} with {len(all_labels):,} entities.")
        
        logger.info("="*50)
        logger.info("   Update process fully completed.")
        logger.info("="*50)
    else:
        logger.info("No new descriptions were added. Local pickle files remain unchanged.")


if __name__ == "__main__":
    main()
