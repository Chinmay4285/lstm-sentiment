import os

# CHANGE THIS to the folder where you want the project
BASE_DIR = r"C:\Users\chinm\Desktop\Projects"  # Example for Windows
# BASE_DIR = "/home/user/projects"    # Example for Linux/Mac

# Project structure definition
structure = {
    "lstm-sentiment": {
        "model.py": "",
        "train.py": "",
        "app.py": "",
        "utils.py": "",
        "templates": {
            "index.html": ""
        },
        "static": {
            "style.css": ""
        },
        "requirements.txt": "",
        "README.md": ""
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)  # Recursively create subfolders
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)  # Empty content

if __name__ == "__main__":
    create_structure(BASE_DIR, structure)
    print(f"✅ Project structure 'lstm-sentiment' created inside: {BASE_DIR}")
