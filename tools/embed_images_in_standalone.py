import base64
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HTML_PATH = os.path.join(ROOT, "orchestrator_overview_standalone.html")

# Map of relative src to filesystem path
IMAGES = {
    "/static/img/poc1_prd_flow.png": os.path.join(ROOT, "static", "img", "poc1_prd_flow.png"),
    "/static/img/poc2_backlog_flow.png": os.path.join(ROOT, "static", "img", "poc2_backlog_flow.png"),
    "/static/img/poc3_delivery_flow.png": os.path.join(ROOT, "static", "img", "poc3_delivery_flow.png"),
    "/static/img/poc3b_requirements_flow.png": os.path.join(ROOT, "static", "img", "poc3b_requirements_flow.png"),
    "/static/img/poc4_data_migration_flow.png": os.path.join(ROOT, "static", "img", "poc4_data_migration_flow.png"),
    "/static/img/poc5_architecture_workbench_flow.png": os.path.join(ROOT, "static", "img", "poc5_architecture_workbench_flow.png"),
}

def to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def main():
    if not os.path.exists(HTML_PATH):
        raise SystemExit(f"HTML not found: {HTML_PATH}")

    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = {}
    for src, path in IMAGES.items():
        if not os.path.exists(path):
            print(f"[WARN] Image missing: {path}")
            continue
        data_url = to_data_url(path)
        replacements[src] = data_url

    # Replace src attributes; also remove onerror fallback for these images (optional)
    updated = html
    for src, data_url in replacements.items():
        # Replace only occurrences in src="..."
        updated = updated.replace(f'src="{src}"', f'src="{data_url}"')

    if updated == html:
        print("[INFO] No changes applied (patterns not found or already inlined).")
        return

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(updated)
    print("[OK] Inlined data URLs into:", HTML_PATH)

if __name__ == "__main__":
    main()
