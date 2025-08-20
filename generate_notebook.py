import os
import argparse
from difflib import SequenceMatcher
from itertools import cycle
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess

def collect_blocks(root, limit=None):
    """Traverse directory and collect text blocks, removing duplicates and similar blocks."""
    blocks = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".md", ".txt", ".py")):
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                blocks.extend(paragraphs)
                if limit and len(blocks) >= limit:
                    blocks = blocks[:limit]
                    break
        if limit and len(blocks) >= limit:
            break
    return simplify_blocks(blocks)

def simplify_blocks(blocks, threshold=0.9):
    """Remove duplicate or highly similar blocks."""
    unique = []
    for block in blocks:
        if not any(SequenceMatcher(None, block, u).ratio() >= threshold for u in unique):
            unique.append(block)
    return unique

def create_story(blocks):
    characters = ["Alice", "Bob", "Charlie", "Dana"]
    images = [f"https://placehold.co/600x400?text=Image+{i}" for i in range(len(blocks))]
    audios = [f"https://example.com/audio{i}.mp3" for i in range(len(blocks))]
    story = []
    char_cycle = cycle(characters)
    for i, block in enumerate(blocks):
        story.append(
            {
                "character": next(char_cycle),
                "text": block,
                "image": images[i],
                "audio": audios[i],
            }
        )
    return story

def build_notebook(story, out_path):
    nb = new_notebook()
    nb.cells.append(new_markdown_cell("# Interactive Story"))
    nb.cells.append(new_code_cell("story = {}".format(repr(story))))
    navigation_code = """
import ipywidgets as widgets
from IPython.display import display, Markdown
index = 0
out = widgets.Output()

def show(i):
    block = story[i]
    md = f"### {block['character']}\\n![]({block['image']})\\n[Audio]({block['audio']})\\n\\n{block['text']}"
    with out:
        out.clear_output()
        display(Markdown(md))

show(index)

def next_block(_):
    global index
    if index < len(story) - 1:
        index += 1
        show(index)

def prev_block(_):
    global index
    if index > 0:
        index -= 1
        show(index)

next_btn = widgets.Button(description='Next')
prev_btn = widgets.Button(description='Previous')
next_btn.on_click(next_block)
prev_btn.on_click(prev_block)
display(widgets.HBox([prev_btn, next_btn]), out)
"""
    nb.cells.append(new_code_cell(navigation_code))
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def export_notebook(nb_path, fmt):
    subprocess.run(["jupyter", "nbconvert", "--to", fmt, nb_path], check=True)

def main():
    parser = argparse.ArgumentParser(description="Generate interactive story notebook")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of blocks")
    parser.add_argument("--export", choices=["html", "pdf"], help="Export notebook format")
    args = parser.parse_args()

    blocks = collect_blocks(args.root, limit=args.limit)
    story = create_story(blocks)
    nb_path = "interactive_story.ipynb"
    build_notebook(story, nb_path)
    if args.export:
        export_notebook(nb_path, args.export)

if __name__ == "__main__":
    main()
