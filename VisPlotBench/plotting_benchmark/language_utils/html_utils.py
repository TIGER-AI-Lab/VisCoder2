import csv
import json
import re
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from .base_handler import LanguageHandler
from .base_utils import (
    remove_ansi_escape,
    remove_bytes_literal, 
    shrink_png_b64_to_under_kb,
    deduplicate_lines,
    extract_fenced_code
)

def csv_to_list_of_dicts(file_path: str) -> list[dict]:
    """
    Convert a CSV file to a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list[dict]: A list of dictionaries representing the CSV data.
    """
    data = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(dict(row))  # Convert each row to a dictionary and append to the list.
    return data

class HTMLHandler(LanguageHandler):
    """Handler for processing HTML code."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config or {})

    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the results of a Jupyter notebook containing plot outputs.

        Args:
            plots_path (Path): Path to the notebook file.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed results with columns:
                - id: Identifier for the code snippet.
                - error: Error message, if any.
                - plots_generated: List of generated plots.
                - has_plot: Boolean indicating if a plot was generated.
        """
        with open(plots_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        plot_results = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            code = cell["source"].lstrip("\n")
            if not code.startswith("# id = "):
                continue

            id_line = code.split("\n")[0].lstrip("# id = ")
            if (id_line.startswith('"') and id_line.endswith('"')) or \
               (id_line.startswith("'") and id_line.endswith("'")):
                id_value = id_line[1:-1]
            else:
                id_value = id_line

            cell_res = {"id": id_value, "error": "", "plots_generated": []}

            images = []
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    traceback_raw = "\n".join(output.get("traceback", []))
                    err = remove_ansi_escape(traceback_raw)
                    err = remove_bytes_literal(err)
                    cell_res["error"] = err  # Extract and clean error messages.
                elif output.output_type == "display_data" and "image/png" in output.data:
                    b64_png = output.data["image/png"]
                    b64_png = shrink_png_b64_to_under_kb(b64_png, max_kb=200, step=0.9, min_side=64)
                    images.append(b64_png)  # Append resized image data.

            cell_res["plots_generated"] = images
            cell_res["has_plot"] = len(images) > 0
            plot_results.append(cell_res)

        return pd.DataFrame(plot_results)

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract HTML plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted HTML code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['html', 'json'],
            remove_stop_tokens=True
        )
        try:
            obj = json.loads(code)
            code = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
        
        return code.strip()
    
    def build_plots(self, dataset: pd.DataFrame, output_path: Path, csv_folder: Path) -> None:
        """
        Build and execute a Jupyter notebook to generate plots.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files.

        Raises:
            CellExecutionError: If the notebook execution fails.
        """
        # Generate all code cells
        cells = dataset.apply(
            self.generate_code, axis=1, args=(csv_folder,)
        ).tolist()
        
        # Build notebook
        self.build_new_nb(cells, output_path)
        
        # Read and execute notebook
        with open(output_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=10,
            interrupt_on_timeout=True,
            allow_errors=True, 
            kernel_name="python3",
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": "."}})
        except CellExecutionError as e:
            print(f"[WARNING] Execution stopped early: {e}\nNotebook will still be saved.")

        with open(output_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        
        print(f"HTML plots notebook saved to {output_path}")

    def build_new_nb(self, blocks: list, plots_nb_path: Path) -> None:
        """
        Save code blocks to a Jupyter notebook.

        Args:
            blocks (list): List of code blocks.
            plots_nb_path (Path): Path to save the notebook.
        """
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_code_cell(block) for block in blocks]

        with open(plots_nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
    

    def generate_code(self, item: pd.Series, csv_folder: Path) -> str:
        """
        Generate executable code for rendering HTML plots.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.
            csv_folder (Path): Path to the folder containing CSV files.

        Returns:
            str: The generated code block.
        """
        item_id = item["id"]
        csv_path = csv_folder / f"{item_id}.csv"
        deduped_code = deduplicate_lines(item["code"], max_repeat=5)
        html_src = deduped_code.strip()

        csv_list_dict = csv_to_list_of_dicts(csv_path)
        json_csv_str = json.dumps(csv_list_dict)
        data_str = "const data = " + json_csv_str.encode("unicode_escape").decode("utf-8")
        html_src = re.sub(r"const\s+data\s*=\s*\[\{html\.csv\}\]\s*;?", data_str, html_src)

        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import os",
            "import re",
            "import json",
            "import asyncio",
            "import nest_asyncio",
            "from io import BytesIO",
            "from PIL import Image, ImageOps",
            "from playwright.async_api import async_playwright",
            "",
            '''
def compute_major_px_ratio(image):
    # Compute the ratio of the most common pixel in the image
    px_count = {}
    for px in image.getdata():
        if px in px_count:
            px_count[px] += 1
        else:
            px_count[px] = 1
    max_px = max(px_count, key=px_count.get)
    return px_count[max_px] / len(image.getdata())

def process_image(image, max_size=(2560, 1440), major_px_threshold=0.98, aspect_ratio_threshold=10, filter_small=True):
    # Resize the image if it exceeds the maximum size
    if image.size[0] * image.size[1] > max_size[0] * max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert alpha channel to white background
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")
    major_px_ratio = compute_major_px_ratio(image)
    if major_px_ratio >= major_px_threshold:
        raise ValueError(f"Image rejected due to high major pixel ratio: {major_px_ratio:.2f}")
    byte_array = image.tobytes()
    width, height = image.size
    aspect_ratio = max(width, height) / min(width, height)

    image_from_byte_array = Image.frombytes("RGB", (width, height), byte_array)
    
    return image_from_byte_array

def extract_html_width(html):
    pattern = r'max-width:\s*(\d+)px'
    match = re.search(pattern, html)
    
    if match:
        return int(match.group(1))
    
    pattern = r'width:\s*(\d+)px'
    match = re.search(pattern, html)
    
    if match:
        return int(match.group(1))
    
    return None
def extract_html_height(html):
    pattern = r'height:\s*(\d+)px'
    match = re.search(pattern, html)
    
    if match: return int(match.group(1))
    else: return None

# Allow nested event loops
nest_asyncio.apply()

async def render_html_async(html, full_page=True, random_width=True):
    try:
        html = html.replace("initial-scale=1.0", "initial-scale=2.0")
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            
            height = extract_html_height(html)
            width = extract_html_width(html)
            if width is None: width = 800
            if height is None: 
                if width is not None and width <= 800:
                    height = 600
                else:
                    height = 800

            page = await browser.new_page(viewport={"width": width, "height": height})
            
            # Collect console messages and errors
            console_errors = []
            page_errors = []
            
            # Filter out WARNING messages
            def handle_console(msg):
                if msg.type == "error" and "WARNING" not in msg.text.upper():
                    console_errors.append(f"Console {msg.type}: {msg.text}")
            
            page.on("console", handle_console)
            page.on("pageerror", lambda error: page_errors.append(f"Page Error: {str(error)}"))
            page.on("requestfailed", lambda request: page_errors.append(f"Request failed: {request.url}"))
            
            await page.set_content(html)
            await page.wait_for_timeout(5000)
            
            screenshot_bytes = await page.screenshot(full_page=full_page)
            await browser.close()
            
            # Only raise exception for real errors
            all_errors = console_errors + page_errors
            if all_errors:
                error_summary = "\\n".join(all_errors)
                raise Exception(f"HTML rendering errors found:\\n{error_summary}")
            
            image = Image.open(BytesIO(screenshot_bytes))
            return image
            
    except Exception as e:
        if "HTML rendering errors found" in str(e):
            raise
        raise Exception(f"Rendering failed: {str(e)}")

# Synchronous wrapper function
def render_html(html, full_page=True, random_width=True):
    return asyncio.run(render_html_async(html, full_page, random_width))
'''.strip(),
        "",
        f"html_code = r'''{html_src}'''",
        "img = render_html(html_code)",
        "img = process_image(img)",
        "display(img)"
        ])

        return "\n".join(code_blocks)
