import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import argparse
import sys

def collect_stats(base_dir: str, stat_filename: str = "benchmark_stat.jsonl") -> Dict[str, List[Dict]]:
    """
    Collect statistics from all language directories.
    
    Args:
        base_dir: Base directory containing language subdirectories
        stat_filename: Name of the statistics file to look for
        
    Returns:
        Dict mapping model names to lists of statistics
    """
    base_path = Path(base_dir)
    language_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # Store statistics for each model
    model_stats = defaultdict(list)

    for lang_dir in language_dirs:
        lang_name = lang_dir.name
        stat_file = lang_dir / stat_filename
        
        if not stat_file.exists():
            # Try to find result files and extract information from them
            result_files = list(lang_dir.glob("results_*.json"))
            if not result_files:
                print(f"[WARNING] No statistics or result files found in {lang_name} directory")
                continue
                
            # Process result files if stat file doesn't exist
            for result_file in result_files:
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)
                    
                    # Extract model name from filename
                    filename = result_file.name
                    model_parts = filename.split('_')
                    if len(model_parts) >= 2:
                        model_name = model_parts[1]  # Assuming format: results_ModelName_language_...
                        
                        # Create synthetic stat entry
                        stat = {
                            'model': model_name,
                            'plotting_lang': lang_name,
                            'scores': extract_scores_from_results(result_data)
                        }
                        model_stats[model_name].append(stat)
                except Exception as e:
                    print(f"[WARNING] Error processing {result_file}: {e}")
            continue
            
        # Process stat file if it exists
        try:
            with open(stat_file, "r") as f:
                for line in f:
                    try:
                        stat = json.loads(line.strip())
                        stat['plotting_lang'] = lang_name
                        model_stats[stat['model']].append(stat)
                    except json.JSONDecodeError:
                        print(f"[WARNING] Error parsing a line in {lang_name}/{stat_filename}")
                        continue
        except Exception as e:
            print(f"[WARNING] Error reading {stat_file}: {e}")
    
    return model_stats

def extract_scores_from_results(result_data):
    """
    Extract score information from result data if available.
    
    Args:
        result_data: Result data from a results JSON file
        
    Returns:
        Dict containing score information or empty dict if not available
    """
    scores = {'vis': {}, 'task': {}}
    
    # This is a placeholder - actual implementation would depend on
    # the structure of the results files
    
    return scores

def print_model_stats_markdown(model_stats: Dict[str, List[Dict]]):
    """
    Print statistics in Markdown table format.
    
    Args:
        model_stats: Dict mapping model names to lists of statistics
    """
    # Define table headers
    headers = ["Model", "Language", "Vis Mean", "Task Mean", "Vis Good", "Task Good"]
    # Set column widths
    widths = [35, 12, 10, 10, 10, 10]
    
    # Print header row
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    print(header_row)
    
    # Print separator row
    separator = "|" + "|".join(f"{'-'*w:^{w+2}}" for w in widths) + "|"
    print(separator)
    
    for model, stats_list in model_stats.items():
        # Get simplified model name
        short_model = model.split('/')[-1]

        # Sort by language name
        stats_list.sort(key=lambda x: x['plotting_lang'])

        # Print statistics for each language
        for stat in stats_list:
            lang = stat['plotting_lang']
            scores = stat.get('scores', {})
            
            # Get vis statistics
            vis_stats = scores.get('vis', {})
            vis_mean = vis_stats.get('mean', 'N/A')
            vis_good = vis_stats.get('good', 'N/A')
            if vis_good != 'N/A':
                vis_good = f"{vis_good:.2f}"
            
            # Get task statistics
            task_stats = scores.get('task', {})
            task_mean = task_stats.get('mean', 'N/A')
            task_good = task_stats.get('good', 'N/A')
            if task_good != 'N/A':
                task_good = f"{task_good:.2f}"
            
            # Format each field
            row = [
                f"{short_model:<{widths[0]}}",
                f"{lang:<{widths[1]}}",
                f"{vis_mean:>{widths[2]-1}} ",
                f"{task_mean:>{widths[4]-1}} ",
                f"{vis_good:>{widths[3]-1}} ",
                f"{task_good:>{widths[5]-1}} "
            ]
            
            # Print row
            print("| " + " | ".join(row) + " |")
        
        # Add separator between different models
        print(separator)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and display judge scores from evaluation results."
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default="eval_results",
        help="Path to eval_results directory or a specific results directory."
    )
    parser.add_argument(
        "--stat_file",
        type=str,
        default="benchmark_stat.jsonl",
        help="Name of the statistics file to look for (default: benchmark_stat.jsonl)."
    )
    args = parser.parse_args()
    
    model_stats = collect_stats(args.input_path, args.stat_file)
    
    if model_stats:
        print_model_stats_markdown(model_stats)
    else:
        print("No statistics found")

if __name__ == "__main__":
    main()