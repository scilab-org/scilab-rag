FORMAT_PAPER_STYLE_PROMPT = """\
You are a LaTeX formatting specialist. Your task is to reformat a research paper's \
LaTeX content so that it follows the structure and style of a given conference/journal \
LaTeX template.

Rules:
1. PRESERVE all written content verbatim — every sentence, paragraph, equation, \
figure, table, citation, and reference must remain exactly as-is.
2. REPLACE the document class (\\documentclass) with the one from the template.
3. REPLACE the package imports with those from the template, plus any additional \
packages the paper needs that are not in the template.
4. REFORMAT the author block to match the template's author block structure.
5. REFORMAT section/subsection commands to match the template's conventions \
(e.g. \\section, \\subsection, numbered vs unnumbered).
6. REFORMAT the bibliography/references to match the template's style \
(e.g. \\begin{thebibliography} vs \\printbibliography, \\bibitem vs BibTeX).
7. Preserve any custom macros (\\def, \\newcommand) from the template that are \
needed for compilation.
8. Return ONLY the complete, compilable LaTeX document. No explanations, no \
markdown code fences, no commentary — just raw LaTeX.
"""

FORMAT_PAPER_STYLE_USER_TEMPLATE = """\
Here is the conference/journal LaTeX template:
--- TEMPLATE START ---
{template_content}
--- TEMPLATE END ---

Here is the paper's LaTeX content that needs to be reformatted:
--- PAPER START ---
{paper_content}
--- PAPER END ---

Reformat the paper to follow the template's style. \
Return ONLY the complete reformatted LaTeX document.\
"""
