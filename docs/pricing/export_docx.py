import os
import pypandoc

def convert_overview_to_docx():
    """
    Converts the pricing overview markdown file to a Word document (.docx).
    This handles tables and LaTeX math formulas automatically.
    """
    # Define absolute paths based on your project structure
    base_dir = r"c:\Users\hoels\var_project"
    input_path = os.path.join(base_dir, "docs", "pricing", "00_overview.md")
    output_path = os.path.join(base_dir, "docs", "pricing", "00_overview.docx")

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