# Quick Start Guide - Compiling the LaTeX Report

This guide helps you compile the Indian Sign Language Detection System report for Overleaf or local LaTeX systems.

## 📋 For Overleaf (Easiest Method)

### Step 1: Create New Project
1. Go to [https://www.overleaf.com](https://www.overleaf.com)
2. Sign up/Log in
3. Click "New Project"
4. Select "Upload Project"

### Step 2: Upload Files
1. Create a ZIP file with all documentation files:
   ```
   Documentation/
   ├── main.tex
   ├── references.bib
   ├── chapters/
   │   ├── chapter2_technologies.tex
   │   ├── chapter3_requirements.tex
   │   ├── chapter4_design.tex
   │   ├── chapter5_pipeline.tex
   │   ├── chapter6_implementation.tex
   │   ├── chapter7_results.tex
   │   └── chapter7_conclusion.tex
   └── diagrams/
       ├── (add your diagram PNG files here)
   ```
2. Upload ZIP to Overleaf

### Step 3: Compile
1. In Overleaf, click "Menu" (top left)
2. Make sure "Compiler" is set to "pdfLaTeX"
3. Click "Recompile" button
4. View PDF in preview pane

### Step 4: Export PDF
1. Click download button (↓) next to preview
2. Select "PDF"
3. Download compiled report

---

## 💻 For Local LaTeX (Windows/Mac/Linux)

### Prerequisites

#### Windows
```bash
# Install MiKTeX (includes LaTeX, pdflatex, bibtex)
# Download from: https://miktex.org/download
# OR use Chocolatey:
choco install miktex
```

#### Mac
```bash
# Install MacTeX (includes full LaTeX)
brew install mactex
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### Compilation Steps

#### Method 1: Using pdflatex (Recommended)
```bash
# Navigate to Documentation folder
cd "path\to\Documentation"

# Step 1: Generate PDF with main.tex
pdflatex -interaction=nonstopmode main.tex

# Step 2: Generate bibliography
bibtex main

# Step 3: Regenerate PDF with bibliography
pdflatex -interaction=nonstopmode main.tex

# Step 4: Second recompile (for references to work)
pdflatex -interaction=nonstopmode main.tex

# Result: main.pdf is created
```

#### Method 2: Using makefile (if available)
```bash
cd Documentation
make
```

#### Method 3: Using batch script (Windows)
Create `compile.bat`:
```batch
@echo off
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo Compilation complete! Check main.pdf
pause
```

Run:
```bash
compile.bat
```

---

## 📂 File Organization

Ensure folder structure:
```
Documentation/
├── main.tex                              (Main document)
├── references.bib                        (Bibliography)
├── chapters/                             (Chapter files)
│   ├── chapter2_technologies.tex
│   ├── chapter3_requirements.tex
│   ├── chapter4_design.tex
│   ├── chapter5_pipeline.tex
│   ├── chapter6_implementation.tex
│   ├── chapter7_results.tex
│   └── chapter7_conclusion.tex
├── diagrams/                             (Diagram images)
│   ├── dfd_placeholder.png
│   ├── er_diagram.png
│   ├── usecase_diagram.png
│   ├── class_diagram.png
│   ├── sequence_diagram.png
│   ├── activity_diagram.png
│   ├── accuracy_graph.png
│   ├── fps_performance.png
│   └── training_loss.png
├── README.md                             (Documentation index)
├── PIPELINE_OVERVIEW.md                  (Pipeline details)
└── COMPILE_GUIDE.md                     (This file)
```

---

## 🎯 Common Commands

### Windows (PowerShell)
```powershell
# Navigate to folder
cd "C:\path\to\Documentation"

# Full compilation
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex

# Generate only (no preview)
pdflatex -interaction=batchmode main.tex
```

### Linux/Mac
```bash
# Navigate to folder
cd /path/to/Documentation

# Full compilation
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# With cleanup
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && \
rm -f *.aux *.log *.out *.toc *.bbl *.blg
```

---

## 🔧 Troubleshooting

### Problem: "File not found" error

**Solution**: Ensure all chapter files are in `chapters/` folder
```bash
# Check files exist
ls chapters/
# Should show: chapter2_technologies.tex, chapter3_requirements.tex, etc.
```

### Problem: Bibliography not showing

**Solution**: Run `bibtex` command between pdflatex runs
```bash
pdflatex main.tex
bibtex main          # Must do this!
pdflatex main.tex
pdflatex main.tex    # Third run important for refs
```

### Problem: Diagram images not showing

**Solution**: Check image format and paths
```
1. Images should be in diagrams/ folder
2. Use PNG format (also supports PDF, JPG)
3. In LaTeX: \includegraphics{diagrams/filename.png}
4. Check filename exactly (case-sensitive on Linux/Mac)
```

### Problem: Too many errors after first run

**Solution**: Clean and restart
```bash
# Remove temporary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.blx

# Then recompile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Problem: "Undefined control sequence" error

**Solution**: Update LaTeX packages
```bash
# Windows (MiKTeX)
mpm --admin --upgrade

# Linux
sudo apt-get install --only-upgrade texlive-latex-base
```

---

## 📊 Compilation Output Files

After successful compilation, you'll have:
```
main.pdf              ← Your final report!
main.aux              ← Auxiliary info (auto)
main.bbl              ← Bibliography list (auto)
main.blg              ← Bibliography log (auto)
main.log              ← Compilation log (auto)
main.out              ← Outline info (auto)
main.toc              ← Table of contents (auto)
```

**You only need**: `main.pdf` (delete others to clean up)

---

## 📖 PDF Options

### Print-Ready Version
```latex
% In main.tex, ensure:
\documentclass[12pt,a4paper]{report}
\usepackage[margin=1in]{geometry}

% Compile for print:
pdflatex main.tex
```

### Digital Version (smaller file)
```bash
# Compress PDF after compilation
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
   -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=main_compressed.pdf main.pdf
```

---

## 🎨 Customization

### Change Title
Edit in `main.tex`:
```latex
\title{\textbf{Indian Sign Language (ISL) Detection System\\ ...}}
\date{\today}
```

### Change Colors
Add to preamble:
```latex
\usepackage{xcolor}
\definecolor{myblue}{RGB}{30, 136, 229}
```

### Add Your Logo
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.3\textwidth]{logo.png}
\end{figure}
```

### Change Font Size
```latex
% In document class:
\documentclass[14pt, a4paper]{report}  % 14pt instead of 12pt
```

---

## ✅ Verification Checklist

- [ ] All chapter files present in `chapters/` folder
- [ ] `references.bib` file exists
- [ ] `main.tex` references all chapters correctly
- [ ] Diagram files in `diagrams/` folder (optional for first compilation)
- [ ] LaTeX compiler installed (pdflatex available)
- [ ] BibTeX tool available
- [ ] All file names match exactly (case-sensitive on Linux/Mac)
- [ ] No special characters in folder paths (avoid spaces if possible)

---

## 📞 Getting Help

### If compilation fails:
1. Check error log: `main.log`
2. Most common: Missing files or citations
3. Try cleaning: Remove `.aux`, `.log`, `.bbl` files
4. Retry compilation

### Online Resources:
- Overleaf Help: https://www.overleaf.com/learn
- LeTeX Tutorials: https://www.latex-project.org/help/documentation/
- Stack Exchange: https://tex.stackexchange.com/

---

## 🎉 Success!

When you see:
```
Output written on main.pdf (XXX pages, YYY bytes).
Transcript written on main.log.
```

Your report is ready! Open `main.pdf` to view the complete report.

---

**Last Updated**: February 16, 2026
**LaTeX Version**: Required pdfLatex
**Bibliography**: BibTeX format
**Total Pages**: 50-70 pages (estimated)
**File Size**: ~5-10 MB (with images)
