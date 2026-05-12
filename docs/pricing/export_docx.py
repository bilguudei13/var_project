import os
from pathlib import Path

import pypandoc

def convert_overview_to_docx():
    """
    Converts the pricing overview markdown file to a Word document (.docx).
    This handles tables and LaTeX math formulas automatically.
    """
    pricing_dir = Path(__file__).resolve().parent
    input_path = os.fspath(pricing_dir / "00_overview.md")
    output_path = os.fspath(pricing_dir / "00_overview.docx")

    if not os.path.exists(input_path):
        print(f"Error: Source file not found at {input_path}")
        return

    print(f"Converting {input_path}...")
    
    try:
        # 'docx' is the target format
        pypandoc.convert_file(input_path, 'docx', outputfile=output_path)
        print(f"Successfully created: {output_path}")
    except RuntimeError as e:
        print(f"Conversion failed. Ensure Pandoc is installed on your system. Error: {e}")

if __name__ == "__main__":
    convert_overview_to_docx()