import fire
from omegaconf import OmegaConf
from plotting_benchmark.benchmark import PlottingBenchmark

def main(
    limit: int | list[int] | None = None,
    language: str = None,  # Can be comma-separated list like "python,latex"
):
    config = OmegaConf.load(f"configs/config_lang_test.yaml")

    if config.run_mode not in ["normal", "self_debug"]:
        raise ValueError(f"Invalid run mode: {config.run_mode}")
    
    supported_languages = ["python", "vegalite", "mermaid", "lilypond", "svg", "asymptote", "latex", "html"]
    if language is None:
        languages_to_run = supported_languages
    else:
        languages_to_run = [lang.strip() for lang in language.split(",")]
        invalid_languages = [lang for lang in languages_to_run if lang not in supported_languages]
        if invalid_languages:
            raise ValueError(f"Unsupported language(s): {invalid_languages}. Supported: {supported_languages}")
    
    # Run benchmark for each language
    for lang in languages_to_run:
        config.plotting_language = lang
        config.paths.out_folder = f"eval_results/{lang}/"
        config.paths.error_rate_file = f"eval_results/{lang}.json"
        
        if config.run_mode == "normal":
            config.paths.bench_stat_filename = "benchmark_stat.jsonl"
        elif config.run_mode == "self_debug":  # self_debug mode
            config.paths.bench_stat_filename = "benchmark_stat_self_debug.jsonl"
            config.debug.output_dir = f"debug_results/{lang}/"
        
        # Initialize and run benchmark
        benchmark = PlottingBenchmark(config=config.copy())
        print(f"[INFO] Using model: {config.model_plot_gen.names[0]}")
            
        benchmark.run_benchmark(
                limit, reuse_results=False, load_intermediate=False, only_stats=False, skip_draw=False, skip_score=True
                # limit, reuse_results=True, load_intermediate=False, only_stats=False, skip_draw=True, skip_score=False
        )

if __name__ == "__main__":
    fire.Fire(main)
