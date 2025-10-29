#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse
import sys

# Define all supported visualization languages
LANGUAGES = ["python", "vegalite", "mermaid", "lilypond", "svg", "asymptote", "latex", "html"]

def load_json_file(file_path: Path):
    """
    Load a single JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dict containing JSON contents or empty dict if file cannot be read
    """
    try:
        with file_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot read JSON file: {file_path} ({e})", file=sys.stderr)
        return {}

def calculate_success_rates(data):
    """
    Calculate execution success rates from result data.
    
    Args:
        data: Dict containing execution results
        
    Returns:
        Dict with language -> model -> metric -> value structure
    """
    all_results = defaultdict(lambda: defaultdict(dict))

    for key, value in data.items():
        if isinstance(value, float) or not isinstance(value, dict):
            continue

        try:
            model_path, lang = key.rsplit("_", 1)
        except ValueError:
            continue
        
        if lang not in LANGUAGES:
            continue

        model_name = Path(model_path).name
        if "total_num" not in value or "execution_error_num" not in value:
            continue

        total_cases = value["total_num"]
        if not isinstance(total_cases, (int, float)) or total_cases <= 0:
            continue

        # Init (global over full set)
        init_err = value.get("execution_error_num", 0)
        all_results[lang][model_name]["Init"] = ((total_cases - init_err) / total_cases) * 100

        # Post attempts (global over full set)
        debug_attempts = value.get("debug_attempts", {})
        for attempt in range(3):  # A0, A1, A2
            attempt_key = f"attempt_{attempt}"
            att = debug_attempts.get(attempt_key)
            if isinstance(att, dict) and "execution_error_num" in att:
                # IMPORTANT: keep denominator = total_cases (global)
                att_err = att["execution_error_num"]
                all_results[lang][model_name][f"Post A{attempt}"] = ((total_cases - att_err) / total_cases) * 100
            else:
                all_results[lang][model_name][f"Post A{attempt}"] = None

    return all_results

def create_formatted_table(results, lang):
    """
    Create a clean pandas DataFrame for a given language,
    ensuring consistent columns and rounding values.
    
    Args:
        results: Dict with execution results
        lang: Language to create table for
        
    Returns:
        Pandas DataFrame with formatted results
    """
    if lang not in results or not results[lang]:
        return pd.DataFrame(columns=["Model", "Init", "Post A0", "Post A1", "Post A2"])

    df = pd.DataFrame.from_dict(results[lang], orient="index")
    df.index.name = "Model"
    df.reset_index(inplace=True)

    expected_columns = ["Model", "Init", "Post A0", "Post A1", "Post A2"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    for col in ["Init", "Post A0", "Post A1", "Post A2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df = df[expected_columns]
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and display execution success rates from evaluation results."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to eval_results directory, a specific language JSON file, or a directory containing JSON files."
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="*",
        default=LANGUAGES,
        help=f"Languages to include (default: {LANGUAGES})."
    )
    args = parser.parse_args()

    root = Path(args.input_path)
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Process input path based on its type
    all_results = defaultdict(lambda: defaultdict(dict))
    
    if root.is_file():
        # Single JSON file
        data = load_json_file(root)
        results = calculate_success_rates(data)
        for lang, models in results.items():
            for model, metrics in models.items():
                all_results[lang][model].update(metrics)
    elif root.is_dir():
        # Directory - could be eval_results or another directory
        if root.name == "eval_results":
            # Process language JSON files in eval_results
            for lang in args.langs:
                lang_json = root / f"{lang}.json"
                if lang_json.exists():
                    data = load_json_file(lang_json)
                    results = calculate_success_rates(data)
                    for l, models in results.items():
                        for model, metrics in models.items():
                            all_results[l][model].update(metrics)
        else:
            # Process all JSON files in directory
            for json_file in root.glob("*.json"):
                data = load_json_file(json_file)
                results = calculate_success_rates(data)
                for lang, models in results.items():
                    for model, metrics in models.items():
                        all_results[lang][model].update(metrics)

    # Print results for each language
    for lang in args.langs:
        if lang in all_results and all_results[lang]:
            print(f"\n=== {lang.upper()} Execution Success Rates (%) ===")
            df = create_formatted_table(all_results, lang)
            print(df.to_markdown(index=False))
            print()
        else:
            print(f"\nNo data available for language: {lang}\n")

if __name__ == "__main__":
    main()