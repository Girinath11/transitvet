# TransitVet manuscript in the MNRAS / RASTI LaTeX layout

- `transitvet.tex` is the manuscript, `transitvet.pdf` the compiled copy (13 pages).
- `figures/` holds the figures, `mnras.cls` and `mnras.bst` are the MNRAS class files (from TeX Live 2023).
- `build_tex.py` rebuilds `transitvet.tex` from `../manuscript.md`. Edit the Markdown, then run
  `python3 build_tex.py && pdflatex transitvet && pdflatex transitvet`. You can also edit `transitvet.tex` directly.

**Overleaf.** Upload `transitvet_overleaf.zip` with New Project > Upload Project. It compiles with pdfLaTeX as it is.

**Submitting to RAS Techniques and Instruments.** RASTI uses its own Overleaf template that follows the MNRAS commands.
Open the RASTI template on Overleaf, paste in everything between `\begin{document}` and `\end{document}`, copy the
title, author and keyword blocks, and upload the `figures/` folder [CHECK against the current RASTI author guidelines].

**Still needed from the team before submission**
1. Affiliation wording, ORCIDs and the corresponding author's e-mail (marked [CHECK] in the author block).
2. The cause of the host-star effect in Section 6 (marked [CHECK] in the text): it is a hypothesis, not tested.
3. The guide's choice of journal; keywords follow the MNRAS/RASTI list format and should be checked against it.
4. Optional: archive the release on Zenodo and add its DOI to the Data availability section.
