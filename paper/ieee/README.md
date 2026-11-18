# TransitVet paper, IEEE conference format

This is the college/IEEE conference version of the paper. It follows the layout of the department's model papers: two columns, Roman-numeral sections, and numbered references.

- `transitvet_ieee.pdf` is the LaTeX build (IEEEtran conference class, A4, 7 pages).
- `transitvet_ieee.docx` is the Word build on the IEEE conference Word template (5 pages, with figures at column width).
- `ieee.md` is the single source for both. Edit it, then run
  `python3 build_ieee.py && pdflatex transitvet_ieee && pdflatex transitvet_ieee`.
- `transitvet_ieee_overleaf.zip` contains the tex, the class and the figures. Upload it to Overleaf with New Project > Upload Project.

The longer journal version (MNRAS/RASTI layout) is in `../journal/rasti/`.

**Fill in before submission:** each author's register number and e-mail, and the guide's designation (all marked [CHECK]). Section VI-F also has one [CHECK]: the cause of the host-star effect has not been tested.
