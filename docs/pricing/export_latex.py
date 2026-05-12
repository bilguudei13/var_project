import os
from pathlib import Path

import pypandoc

def convert_overview_to_latex():
    """
    Converts the pricing overview markdown file to a LaTeX document (.tex).
    This handles tables and LaTeX math formulas automatically.
    """
    pricing_dir = Path(__file__).resolve().parent
    input_path = os.fspath(pricing_dir / "00_overview.md")
    output_path = os.fspath(pricing_dir / "00_overview.tex")

    if not os.path.exists(input_path):
        print(f"Error: Source file not found at {input_path}")
        return

    print(f"Converting {input_path} to LaTeX...")
    
    try:
        # 'latex' is the target format
        # You might want to add --standalone to create a full LaTeX document
        # or just output the body content. For direct copy-paste, body is often preferred.
        pypandoc.convert_file(input_path, 'latex', outputfile=output_path, extra_args=['--standalone'])
        print(f"Successfully created: {output_path}")
    except RuntimeError as e:
        print(f"Conversion failed. Ensure Pandoc is installed on your system. Error: {e}")

if __name__ == "__main__":
    convert_overview_to_latex()