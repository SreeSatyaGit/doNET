import os

def add_header(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    
    comment_styles = {
        '.py': '# {}',
        '.r': '# {}',
        '.yml': '# {}',
        '.yaml': '# {}',
        '.txt': '# {}',
        '.gitignore': '# {}',
        '.ini': '# {}',
        '.md': '<!-- {} -->',
        '': '# {}' # for files like LICENSE if I touch them, but I'll filter
    }
    
    if ext == '.ipynb':
        return # Skip notebooks for now to avoid breaking JSON structure
        
    if filename == 'LICENSE':
        return
        
    style = comment_styles.get(ext, '# {}')
    header = style.format(filename) + '\n'
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check if header already exists to avoid duplication
        if content.startswith(header):
            print(f"Skipping {filepath}, header already present.")
            return

        with open(filepath, 'w') as f:
            f.write(header + content)
        print(f"Added header to {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    files = [
        ".github/workflows/ci.yml",
        ".gitignore",
        "CLAUDE.md",
        "LICENSE",
        "R/AMLTirated.R",
        "R/DataProcessing.R",
        "R/GSM6805326.R",
        "R/RefMapping.R",
        "R/WNN_Mapping.R",
        "README.md",
        "environment.yml",
        "pytest.ini",
        "requirements.txt",
        "research/prepare.py",
        "research/program.md",
        "research/pyproject.toml",
        "research/train.py",
        "run_experiment.py",
        "scripts/__init__.py",
        "scripts/data_provider/__init__.py",
        "scripts/data_provider/data_preprocessing.py",
        "scripts/data_provider/graph_data_builder.py",
        "scripts/data_provider/synthetic_citeseq.py",
        "scripts/model/__init__.py",
        "scripts/model/doNET.py",
        "scripts/trainer/gat_trainer.py",
        "scripts/visualizations.py"
    ]
    
    for f in files:
        if os.path.exists(f):
            add_header(f)
