#!/usr/bin/env python3
"""
Setup script for BAN 501 Course Companion MkDocs site.

Copies markdown files from the Course Companion directory, transforms image paths,
and generates MkDocs configuration files.

Usage:
    pixi run python setup_mkdocs.py --source-dir "/path/to/Course Companion"
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

# File mapping: source filename -> (destination subdirectory, destination filename)
FILE_MAPPING: Dict[str, Tuple[str, str]] = {
    "Module 1 - Foundations of Machine Learning.md": ("modules", "01-foundations.md"),
    "Module 2 - Regression.md": ("modules", "02-regression.md"),
    "Module 3 - Classification.md": ("modules", "03-classification.md"),
    "Module 4 - Ensemble Methods.md": ("modules", "04-ensemble-methods.md"),
    "Module 5 - Unsupervised Learning.md": ("modules", "05-unsupervised.md"),
    "Module 6 - Neural Networks.md": ("modules", "06-neural-networks.md"),
    "Module 7 - Computer Vision.md": ("modules", "07-computer-vision.md"),
    "Module 8 - NLP.md": ("modules", "08-nlp.md"),
    "Module 9 - Interpretability.md": ("modules", "09-interpretability.md"),
    "Module 10 - Ethics and Deployment.md": ("modules", "10-ethics-deployment.md"),
    "Deep Dive - Neural Networks as Universal Approximators.md": (
        "appendices",
        "universal-approximators.md",
    ),
    "Deep Dive - CNN Architecture.md": ("appendices", "cnn-architecture.md"),
    "Deep Dive - Transformer Architecture.md": ("appendices", "transformer-architecture.md"),
}

SITE_CONFIG = {
    "name": "BAN 501 Course Companion",
    "description": "Predictive Modeling - Course Companion",
    "author": "Nick Freeman",
    "url": "https://nkfreeman-teaching.github.io/BAN-501-course-companion/",
    "repo_url": "https://github.com/nkfreeman-teaching/BAN-501-course-companion",
    "repo_name": "nkfreeman-teaching/BAN-501-course-companion",
}


# ============================================================================
# FILE TRANSFORMATION
# ============================================================================


def transform_image_paths(content: str) -> str:
    """
    Transform image paths from Obsidian format to MkDocs format.

    Handles both URL-encoded and non-encoded paths:
    - ../Slides%20Assets/module1/file.png -> ../assets/module1/file.png
    - ../Slides Assets/module1/file.png -> ../assets/module1/file.png
    """
    # Pattern for URL-encoded spaces
    content = re.sub(
        r"\.\./Slides%20Assets/",
        "../assets/",
        content,
    )

    # Pattern for literal spaces
    content = re.sub(
        r"\.\./Slides Assets/",
        "../assets/",
        content,
    )

    return content


def process_markdown_file(
    source_path: Path,
    dest_path: Path,
) -> None:
    """Read a markdown file, transform image paths, and write to destination."""
    content = source_path.read_text(encoding="utf-8")
    transformed = transform_image_paths(content)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(transformed, encoding="utf-8")


# ============================================================================
# COPY FUNCTIONS
# ============================================================================


def copy_markdown_files(
    source_dir: Path,
    output_dir: Path,
) -> list[str]:
    """
    Copy and transform all markdown files according to FILE_MAPPING.

    Returns list of copied files for logging.
    """
    docs_dir = output_dir / "docs"
    copied_files = []

    for source_name, (subdir, dest_name) in FILE_MAPPING.items():
        source_path = source_dir / source_name
        dest_path = docs_dir / subdir / dest_name

        if source_path.exists():
            process_markdown_file(
                source_path=source_path,
                dest_path=dest_path,
            )
            copied_files.append(f"{subdir}/{dest_name}")
        else:
            print(f"  Warning: Source file not found: {source_name}")

    return copied_files


def copy_assets(
    slides_assets_dir: Path,
    output_dir: Path,
) -> int:
    """
    Copy Slides Assets directory to docs/assets/.

    Returns count of files copied.
    """
    dest_assets = output_dir / "docs" / "assets"

    if slides_assets_dir.exists():
        # Remove existing assets if present
        if dest_assets.exists():
            shutil.rmtree(dest_assets)

        shutil.copytree(
            src=slides_assets_dir,
            dst=dest_assets,
        )

        # Count image files
        return len(list(dest_assets.rglob("*.png")))
    else:
        print(f"  Warning: Slides Assets directory not found: {slides_assets_dir}")
        return 0


# ============================================================================
# CONFIGURATION FILE GENERATORS
# ============================================================================


def generate_index_md() -> str:
    """Generate the home page content."""
    return """# BAN 501: Predictive Modeling

## Course Companion

Welcome to the BAN 501 Course Companion. This resource provides comprehensive coverage of predictive modeling concepts, from foundational machine learning principles to advanced deep learning applications.

## How to Use This Companion

- **Sequential reading**: Work through modules 1-10 in order for a complete learning path
- **Reference**: Jump to specific topics using the navigation sidebar
- **Search**: Use the search bar to find specific concepts or terms
- **Deep dives**: Explore appendices for extended technical discussions

## Module Overview

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| 1 | Foundations | Train/test splits, bias-variance tradeoff, evaluation metrics |
| 2 | Regression | Linear regression, gradient descent, regularization |
| 3 | Classification | Logistic regression, decision boundaries, ROC/AUC |
| 4 | Ensemble Methods | Bagging, boosting, random forests, XGBoost |
| 5 | Unsupervised Learning | Clustering, dimensionality reduction, PCA |
| 6 | Neural Networks | Perceptrons, backpropagation, deep learning |
| 7 | Computer Vision | CNNs, image classification, transfer learning |
| 8 | NLP | Text processing, embeddings, transformers |
| 9 | Interpretability | SHAP, LIME, feature importance |
| 10 | Ethics & Deployment | Fairness, model deployment, monitoring |

## Prerequisites

This companion assumes familiarity with:

- Basic Python programming
- Introductory statistics (mean, variance, distributions)
- Linear algebra fundamentals (matrices, vectors)

## Appendices

The appendices provide deeper technical explorations:

- **Universal Approximators**: Why neural networks can learn any function
- **CNN Architecture**: Detailed breakdown of convolutional networks
- **Transformer Architecture**: The architecture behind modern NLP
"""


def generate_mkdocs_yml() -> str:
    """Generate mkdocs.yml configuration."""
    return f"""site_name: {SITE_CONFIG["name"]}
site_description: {SITE_CONFIG["description"]}
site_author: {SITE_CONFIG["author"]}
site_url: {SITE_CONFIG["url"]}

repo_url: {SITE_CONFIG["repo_url"]}
repo_name: {SITE_CONFIG["repo_name"]}

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
  icon:
    repo: fontawesome/brands/github

nav:
  - Home: index.md
  - Modules:
    - "1. Foundations of ML": modules/01-foundations.md
    - "2. Regression": modules/02-regression.md
    - "3. Classification": modules/03-classification.md
    - "4. Ensemble Methods": modules/04-ensemble-methods.md
    - "5. Unsupervised Learning": modules/05-unsupervised.md
    - "6. Neural Networks": modules/06-neural-networks.md
    - "7. Computer Vision": modules/07-computer-vision.md
    - "8. NLP": modules/08-nlp.md
    - "9. Interpretability": modules/09-interpretability.md
    - "10. Ethics & Deployment": modules/10-ethics-deployment.md
  - Appendices:
    - "Neural Networks as Universal Approximators": appendices/universal-approximators.md
    - "CNN Architecture": appendices/cnn-architecture.md
    - "Transformer Architecture": appendices/transformer-architecture.md

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - tables
  - toc:
      permalink: true

extra_javascript:
  - javascripts/mathjax.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

extra_css:
  - stylesheets/extra.css

plugins:
  - search
"""


def generate_mathjax_js() -> str:
    """Generate MathJax configuration."""
    return r"""window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
"""


def generate_extra_css() -> str:
    """Generate custom CSS styles."""
    return """/* Custom styles for BAN 501 Course Companion */

/* Center images by default */
img {
  display: block;
  margin: 0 auto;
  max-width: 100%;
}

/* Code block styling */
.highlight pre {
  padding: 1em;
}

/* Table improvements */
table {
  width: 100%;
}
"""


def generate_deploy_yml() -> str:
    """Generate GitHub Actions deployment workflow."""
    return """name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material

      - name: Build site
        run: mkdocs build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""


def generate_requirements_txt() -> str:
    """Generate requirements.txt for GitHub Actions."""
    return """mkdocs>=1.6
mkdocs-material>=9.5
"""


# ============================================================================
# MAIN SETUP FUNCTION
# ============================================================================


def create_directory_structure(output_dir: Path) -> None:
    """Create the MkDocs directory structure."""
    directories = [
        output_dir / "docs" / "modules",
        output_dir / "docs" / "appendices",
        output_dir / "docs" / "javascripts",
        output_dir / "docs" / "stylesheets",
        output_dir / ".github" / "workflows",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def write_config_files(output_dir: Path) -> None:
    """Write all configuration files."""
    files = {
        output_dir / "docs" / "index.md": generate_index_md(),
        output_dir / "docs" / "javascripts" / "mathjax.js": generate_mathjax_js(),
        output_dir / "docs" / "stylesheets" / "extra.css": generate_extra_css(),
        output_dir / "mkdocs.yml": generate_mkdocs_yml(),
        output_dir / ".github" / "workflows" / "deploy.yml": generate_deploy_yml(),
        output_dir / "requirements.txt": generate_requirements_txt(),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")


def setup_mkdocs(
    source_dir: Path,
    output_dir: Path,
) -> None:
    """Main setup function that orchestrates the entire process."""
    print(f"Setting up MkDocs in: {output_dir}")
    print(f"Source directory: {source_dir}")
    print()

    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create directory structure
    print("Creating directory structure...")
    create_directory_structure(output_dir)

    # Copy and transform markdown files
    print("Copying and transforming markdown files...")
    copied_files = copy_markdown_files(
        source_dir=source_dir,
        output_dir=output_dir,
    )
    print(f"  Copied {len(copied_files)} markdown files")

    # Copy assets
    print("Copying Slides Assets...")
    slides_assets_dir = source_dir.parent / "Slides Assets"
    asset_count = copy_assets(
        slides_assets_dir=slides_assets_dir,
        output_dir=output_dir,
    )
    print(f"  Copied {asset_count} image files")

    # Write configuration files
    print("Writing configuration files...")
    write_config_files(output_dir)

    print()
    print("Setup complete!")
    print()
    print("Next steps:")
    print("  1. Preview locally: pixi run serve")
    print("  2. Commit and push to GitHub")
    print("  3. Enable GitHub Pages (Settings -> Pages -> Source: GitHub Actions)")
    print()
    print(f"Site will be available at: {SITE_CONFIG['url']}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup BAN 501 Course Companion MkDocs site",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_mkdocs.py --source-dir "/path/to/Course Companion"
  python setup_mkdocs.py -s ~/Documents/vault/Teaching/BAN\\ 501/Course\\ Companion
        """,
    )

    parser.add_argument(
        "--source-dir",
        "-s",
        type=Path,
        required=True,
        help="Path to the Course Companion source directory",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = Path(__file__).parent.resolve()

    setup_mkdocs(
        source_dir=source_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
