# BAN 501 Course Companion

A static website for the BAN 501: Predictive Modeling course, built with MkDocs and deployed via GitHub Actions to GitHub Pages.

**Live site:** https://nkfreeman-teaching.github.io/BAN-501-course-companion/

## Repository Structure

```
BAN-501-course-companion/
├── docs/                          # Source content
│   ├── index.md                   # Home page
│   ├── modules/                   # Course modules (1-10)
│   │   ├── 01-foundations.md
│   │   ├── 02-regression.md
│   │   ├── 03-classification.md
│   │   ├── 04-ensemble-methods.md
│   │   ├── 05-unsupervised.md
│   │   ├── 06-neural-networks.md
│   │   ├── 07-computer-vision.md
│   │   ├── 08-nlp.md
│   │   ├── 09-interpretability.md
│   │   └── 10-ethics-deployment.md
│   ├── appendices/                # Deep dive topics
│   │   ├── universal-approximators.md
│   │   ├── cnn-architecture.md
│   │   ├── transformer-architecture.md
│   │   └── surprising-phenomena.md
│   ├── assets/                    # Images organized by module
│   ├── javascripts/
│   │   └── mathjax.js             # MathJax configuration
│   └── stylesheets/
│       └── extra.css              # Custom styles
├── site/                          # Built HTML output (auto-generated)
├── .github/workflows/
│   └── deploy.yml                 # GitHub Actions deployment
├── mkdocs.yml                     # MkDocs configuration
├── pixi.toml                      # Pixi environment config
├── setup_mkdocs.py                # Sync script for Obsidian vault
└── requirements.txt               # Pip dependencies (for CI)
```

## Technologies

- **MkDocs** with **Material** theme for site generation
- **MathJax 3** for LaTeX math rendering
- **GitHub Actions** for automated deployment to GitHub Pages
- **Pixi** for local environment management

## Local Development

### Prerequisites

Install [Pixi](https://pixi.sh/) for environment management.

### Commands

```bash
# Start local development server with live reload
pixi run serve

# Build the site locally
pixi run build

# Deploy manually to GitHub Pages (usually not needed)
pixi run deploy
```

The development server runs at `http://127.0.0.1:8000/` by default.

## Updating the Deployed Site

The site automatically deploys when changes are pushed to the `main` branch.

### Workflow

1. **Edit content** in the `docs/` directory
   - Module content: `docs/modules/`
   - Appendices: `docs/appendices/`
   - Images: `docs/assets/`

2. **Preview locally** (optional but recommended)
   ```bash
   pixi run serve
   ```

3. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

4. **Automatic deployment** - GitHub Actions will:
   - Build the site with MkDocs
   - Deploy to GitHub Pages
   - Changes appear at the live URL within a few minutes

### Adding New Pages

1. Create a new `.md` file in the appropriate directory under `docs/`

2. Add an entry to the `nav` section in `mkdocs.yml`:
   ```yaml
   nav:
     - Modules:
       - "New Module Title": modules/new-module.md
   ```

3. Commit and push

### Adding Images

1. Place images in `docs/assets/` (organize by module if needed)
2. Reference in markdown:
   ```markdown
   ![Alt text](../assets/module1/image.png)
   ```

## Content Sync from Obsidian

If authoring content in an Obsidian vault, use the sync script:

```bash
pixi run sync
```

This copies markdown files from the configured source directory and transforms paths for web publishing. The source directory is configured in `pixi.toml`.

## Math Equations

The site supports LaTeX math via MathJax:

- **Inline math:** `\( x^2 + y^2 = z^2 \)`
- **Display math:** Use `$$...$$` blocks with blank lines before and after

```markdown
Some text before.

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$

Some text after.
```

## Configuration

- **Site metadata:** `mkdocs.yml` (name, description, theme, navigation)
- **Custom styles:** `docs/stylesheets/extra.css`
- **MathJax setup:** `docs/javascripts/mathjax.js`
- **Build dependencies:** `pixi.toml` (local) and `requirements.txt` (CI)
