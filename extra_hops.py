import json
import pandas as pd
import os
import random
from preprocess import DATA_PATHS, normalize_documents, build_prompt

def extract_hops_data(seed=42):
    dataset_name = 'HotpotQA'
    split_type = 'train'
    
    # Get source path from preprocess config
    source_path = DATA_PATHS.get(dataset_name, {}).get(split_type)
    if not source_path:
        print(f"Error: No path configured for dataset '{dataset_name}' split '{split_type}'")
        return

    print(f"Reading from: {source_path}")
    
    # Store raw records by hop
    # Note: we keep a stable global index for each record to guarantee non-overlap
    # between SFT samples and RL samples even if records have duplicated contents.
    records_by_hop = {
        '2': [],
        '3': [],
        '4': []
    }
    
    try:
        global_idx = 0
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    hop = None
                    if dataset_name == 'HotpotQA':
                        supporting_facts = record.get('supporting_facts', [])
                        hop = str(len(supporting_facts))
                    else:
                        rec_id = record.get('id', '')
                        if rec_id:
                            hop = rec_id[0]

                    if hop in records_by_hop:
                        records_by_hop[hop].append((global_idx, record))
                    global_idx += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {source_path}")
        return

    print("Records found:")
    for hop, recs in records_by_hop.items():
        print(f"  {hop}-hop: {len(recs)}")

    # Process and save SFT samples for each hop; then sample again from the remaining
    # records to build an RL training set (train_rl.parquet).
    output_dir = os.path.join('data', dataset_name, 'hops_split')
    os.makedirs(output_dir, exist_ok=True)
    
    random.seed(seed)
    target_count = 5000
    all_rl_rows = []
    
    for hop, records in records_by_hop.items():
        # ---- SFT sampling (keep existing behavior as much as possible) ----
        if len(records) < target_count:
            print(f"Warning: Only found {len(records)} records for {hop}-hop (requested {target_count}). Using all available for SFT.")
            selected_records = records
        else:
            selected_records = random.sample(records, target_count)

        sft_selected_idx = {idx for idx, _ in selected_records}
        remaining_records = [(idx, rec) for idx, rec in records if idx not in sft_selected_idx]

        # ---- RL sampling from remaining (no overlap with SFT) ----
        if len(remaining_records) < target_count:
            print(
                f"Warning: Only found {len(remaining_records)} remaining records for {hop}-hop "
                f"(requested {target_count}). Using all available for RL."
            )
            rl_selected_records = remaining_records
        else:
            rl_selected_records = random.sample(remaining_records, target_count)

        # Safety check: guarantee non-overlap
        rl_selected_idx = {idx for idx, _ in rl_selected_records}
        overlap = sft_selected_idx.intersection(rl_selected_idx)
        if overlap:
            raise RuntimeError(f"Found overlap between SFT and RL selections for hop={hop}: {len(overlap)}")

        # ---- Build SFT parquet for this hop (existing format + optional hop) ----
        processed_data = []
        for _, record in selected_records:
            query = record.get('question', '')
            # Match preprocess.py logic for answer
            ground_truth = record.get('answer', '')
            
            normalized_docs = normalize_documents(record, dataset_name)
            prompt = build_prompt(query, normalized_docs, record, dataset_name)
            
            processed_data.append({
                "prompt": prompt,
                "query": query,
                "ground_truth": ground_truth,
                # "id": record.get('id'), # Optional: keep ID if needed
                # Keep hop info for debugging/analysis (doesn't break downstream parquet readers)
                "hop": int(hop)
            })
            
        output_path = os.path.join(output_dir, f'{hop}hop_1000.parquet')
        df = pd.DataFrame(processed_data)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")

        # ---- Append RL rows (aggregated to train_rl.parquet) ----
        for _, record in rl_selected_records:
            query = record.get('question', '')
            ground_truth = record.get('answer', '')
            normalized_docs = normalize_documents(record, dataset_name)
            prompt = build_prompt(query, normalized_docs, record, dataset_name)
            all_rl_rows.append({
                "prompt": prompt,
                "query": query,
                "ground_truth": ground_truth,
                "hop": int(hop),
            })

    # Save aggregated RL parquet
    rl_output_path = os.path.join('data', dataset_name, 'train_rl.parquet')
    os.makedirs(os.path.dirname(rl_output_path), exist_ok=True)
    rl_df = pd.DataFrame(all_rl_rows)
    rl_df.to_parquet(rl_output_path, index=False)
    print(f"Saved {len(rl_df)} RL records to {rl_output_path}")

if __name__ == "__main__":
    extract_hops_data()

