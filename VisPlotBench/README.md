# VisPlotBench Evaluation Framework

VisPlotBench is an evaluation framework built upon [PandasPlotBench](https://github.com/JetBrains-Research/PandasPlotBench) and [VisCoder](https://github.com/TIGER-AI-Lab/VisCoder/tree/main/eval), extended with a **self-debug evaluation mode** to assess large language models' capabilities in generating visualization code across multiple languages.

## 🚀 Quick Start

### Setup

```bash
conda create -n visplotbench python=3.10 -y
conda activate visplotbench

cd VisPlotBench
pip install -r requirements.txt
```

Depending on the visualization languages you want to evaluate, you may need to install additional dependencies:

1. **LaTeX**: Installation depends on your operating system. Refer to the [official LaTeX website](https://www.latex-project.org/get/) for details.

   For Ubuntu, you can use:
   ```bash
   sudo apt update
   sudo apt install texlive-full
   ```

2. **Mermaid**: 
   ```bash
   mkdir plotting_benchmark/language_utils/env
   cd plotting_benchmark/language_utils/env
   npm init -y 
   npm install @mermaid-js/mermaid-cli
   ```
   For more details, refer to [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli)

3. **HTML**:
   ```bash
   pip install playwright
   playwright install
   ```

4. **Asymptote**:
   ```bash
   sudo apt update
   sudo apt install asymptote
   ```
   For more details, refer to [Asymptote website](https://asymptote.sourceforge.io/)

5. **LilyPond**:
   ```bash
   sudo apt update
   sudo apt install lilypond
   ```
   For more details, refer to [LilyPond website](https://lilypond.org/)

**⚠️ Recommendation**: After setting up the environment, run the examples in `env_test.ipynb` to verify that your environment is correctly configured for all languages and libraries.

### Basic Usage

#### Running Evaluations

The main entry point for running evaluations is `run_benchmark.py`. Key parameters include:

- **limit**: Controls the number of samples to evaluate (integer, list, or None)
- **language**: Specifies the visualization language(s) to evaluate (single or comma-separated)
- **config_file**: Specifies the configuration file path
- **run_mode**: Specifies the evaluation mode (normal or self_debug)

Other important parameters for the `run_benchmark` method:
- **reuse_results**: If `True`, reuses existing results without generating new visualizations
- **load_intermediate**: If `True`, loads from intermediate results file for handling interrupted evaluations
- **only_stats**: If `True`, only calculates statistics without running the full evaluation
- **skip_draw**: If `True`, skips the drawing step
- **skip_score**: If `True`, skips the scoring step

Example command:
```bash
python run_benchmark.py \
    --limit=10 \
    --language=python,svg \
    --config_file="configs/config_visplotbench.yaml" \
    --run_mode=normal
```

Parameter details:
- **limit** parameter controls the evaluation scope:
  - **Integer**: e.g., `10` to randomly select 10 samples for evaluation
  - **List**: e.g., `[0,1,2,3,4,5,6,7]` to evaluate only the specified sample IDs
  - **None**: Evaluate all samples without limitation

- **language** parameter sets the languages to evaluate:
  - Can be a single language (e.g., `python`)
  - Can be a comma-separated list of languages (e.g., `python,latex,svg`)
  - When not provided, evaluates all supported languages

- **config_file** parameter specifies the configuration file, defaults to `configs/config_visplotbench.yaml`

- **run_mode** parameter specifies the evaluation mode, defaults to `normal`, set to `self_debug` to enable self-debugging mode

**Recommendation**: For initial runs, use the `--skip_score=True` option to skip the scoring step. This allows you to quickly generate and review the output in Jupyter Notebook files and results JSON files. If you're only interested in execution pass rates, skipping scoring avoids the high cost associated with using GPT-4o for judging.

#### Configuration

Key configuration values are set in YAML files:
- **run_mode**: Determines the evaluation mode (`normal` or `self_debug`)
- **plotting_language**: Specifies the visualization language to use
- **debug.top_k**: Specifies the maximum number of attempts to consider in self-debug mode
- **model_plot_gen.names**: Defines the list of models to use for code generation
- **model_judge.name**: Defines the model to use for scoring

Key configuration files in the `configs/` directory:
- `config_visplotbench.yaml`: Main configuration file supporting multiple models and languages
- `config_lang_test.yaml`: Configuration file for single language testing

## 📁 Key Files Structure

```bash
── configs/                     # Configuration files
├── plotting_benchmark/         # Core evaluation framework
│ ├── language_utils/           # Utilities for various visualization languages
│ ├── generation_engines/       # Model generation engines
│ ├── benchmark.py              # Main evaluation class
│ ├── vis_generator.py          # Visualization generation and execution
│ ├── vis_judge.py              # Scoring tools
│ └── debug_utils.py            # Self-debugging utilities
│
├── run_benchmark.py            # Main evaluation entry point
├── statistic_tool/             # Statistical analysis tools
└── env_test.ipynb              # Environment testing file
```

## Running Model Evaluations

- **Single Language Test**:
  ```bash
  python run_benchmark.py \
    --language=python \
    --limit=10
  ```

- **Multi-Model Evaluation**: Set all model names in `config_visplotbench.yaml`:
  ```bash
  python run_benchmark.py \
    --run_mode=self_debug
  ```
  When `--language` is not specified, evaluates all supported languages

## 📈 Output Results

- **`eval_results/`**: Main directory for evaluation results
  - **Language-specific directories** (e.g., `python/`, `latex/`):
    - `plots_{model}_{language}_{id}.ipynb`: Jupyter Notebooks containing generated code and execution results
    - `results_{model}_{language}_{descriptor}_{id}.json`: Detailed evaluation results for each model and language combination
    - `benchmark_stat.jsonl` or `benchmark_stat_self_debug.jsonl`: Summary statistics for each evaluation run
  - **Language summary files** (e.g., `python.json`, `latex.json`): Execution error statistics for each model and language

- **`debug_results/`**: Contains detailed self-debug attempts in Jupyter Notebook format, documenting each step and result of the self-debugging process

## Execution Pass Rate Statistics

To print execution success rates from the evaluation results, use the `statistic_tool/print_exec_pass.py` script:

```bash
# Process all languages in the eval_results directory
python statistic_tool/print_exec_pass.py eval_results
```

## Judge Score Statistics

To print judge scores from the evaluation results, use the `statistic_tool/print_judge_score.py` script:

```bash
# Process all languages in the eval_results directory
python statistic_tool/print_judge_score.py eval_results
```

These tools output statistics in a Markdown table format for easy review and inclusion in documentation.

## Acknowledgments

This evaluation framework is built upon [PandasPlotBench](https://github.com/JetBrains-Research/PandasPlotBench) and [VisCoder](https://github.com/TIGER-AI-Lab/VisCoder), both licensed under the Apache License 2.0. We have extended its capabilities with a self-debug evaluation mode to enhance the assessment of LLM visualization code generation and error correction abilities.

For more details on the original frameworks and their licensing, please refer to the [PandasPlotBench repository](https://github.com/JetBrains-Research/PandasPlotBench) and [VisCoder repository](https://github.com/TIGER-AI-Lab/VisCoder).