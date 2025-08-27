import json

def verify_notebook(path):
    """
    Reads a Jupyter notebook, extracts all code cells,
    and executes them to check for errors.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook file not found at '{path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the notebook file at '{path}'")
        return

    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']

    print(f"Found {len(code_cells)} code cells to execute from '{path}'.")

    # Combine all code cells into a single script
    full_script = ""
    for cell_source in code_cells:
        full_script += "\n".join(line for line in cell_source if not line.strip().startswith('%'))
        full_script += "\n\n"

    try:
        # Execute the script
        exec(full_script, globals())
        print("Verification script completed successfully!")
    except Exception as e:
        print(f"An error occurred during notebook execution: {e}")
        # Optionally, re-raise the exception if you want the script to fail
        # raise e

# --- Main execution ---
if __name__ == "__main__":
    notebook_path = "Numpy/Numpy.ipynb"
    verify_notebook(notebook_path)
