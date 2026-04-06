"""
LaTeX validation utilities — programmatic checks that don't need an LLM.

Uses pylatexenc for syntax validation and regex for citation / label extraction.
Falls back gracefully if pylatexenc is not installed.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import pylatexenc; degrade gracefully if missing.
try:
    from pylatexenc.latexwalker import LatexWalker, LatexWalkerError
    HAS_PYLATEXENC = True
except ImportError:
    HAS_PYLATEXENC = False
    logger.warning("pylatexenc not installed — LaTeX syntax checks disabled")


@dataclass
class LatexIssue:
    """A single issue found during validation."""
    type: str          # "syntax" | "citation" | "label"
    description: str
    severity: str      # "error" | "warning"
    line: Optional[int] = None


@dataclass
class LatexValidationResult:
    """Aggregate result of programmatic LaTeX checks."""
    is_valid: bool = True
    issues: list[LatexIssue] = field(default_factory=list)

    def add(self, issue: LatexIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.is_valid = False


# ── Public API ───────────────────────────────────────────────────────────

def validate_latex_syntax(content: str) -> LatexValidationResult:
    """
    Run programmatic LaTeX checks:
    1. pylatexenc parse (catches unmatched environments, bad commands)
    2. Brace balance check
    3. Environment matching (\\begin/\\end pairs)
    """
    result = LatexValidationResult()

    # 1. pylatexenc parse
    if HAS_PYLATEXENC:
        try:
            walker = LatexWalker(content)
            walker.get_latex_nodes()
        except LatexWalkerError as exc:
            result.add(LatexIssue(
                type="syntax",
                description=f"LaTeX parse error: {exc}",
                severity="error",
            ))

    # 2. Brace balance
    _check_brace_balance(content, result)

    # 3. Environment matching
    _check_environment_matching(content, result)

    return result


def extract_citations(content: str) -> list[str]:
    """Extract all \\cite{...} keys from LaTeX content."""
    # Matches \cite{key1,key2}, \citep{key}, \citet{key}, etc.
    pattern = r"\\cite[tp]?\{([^}]+)\}"
    keys: list[str] = []
    for match in re.finditer(pattern, content):
        raw = match.group(1)
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    return keys


def extract_labels(content: str) -> list[str]:
    """Extract all \\label{...} values."""
    return re.findall(r"\\label\{([^}]+)\}", content)


def extract_refs(content: str) -> list[str]:
    """Extract all \\ref{...} and \\eqref{...} values."""
    return re.findall(r"\\(?:eq)?ref\{([^}]+)\}", content)


def check_ref_integrity(content: str) -> list[LatexIssue]:
    """Check that every \\ref{X} has a corresponding \\label{X}."""
    labels = set(extract_labels(content))
    refs = extract_refs(content)
    issues: list[LatexIssue] = []
    for ref in refs:
        if ref not in labels:
            issues.append(LatexIssue(
                type="label",
                description=f"\\ref{{{ref}}} has no matching \\label{{{ref}}} in this section",
                severity="warning",
            ))
    return issues


# ── Internal helpers ─────────────────────────────────────────────────────

def _check_brace_balance(content: str, result: LatexValidationResult) -> None:
    """Check that curly braces are balanced (ignoring escaped braces)."""
    depth = 0
    for i, ch in enumerate(content):
        if ch == "{" and (i == 0 or content[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or content[i - 1] != "\\"):
            depth -= 1
        if depth < 0:
            result.add(LatexIssue(
                type="syntax",
                description="Unexpected closing brace '}' without matching '{'",
                severity="error",
            ))
            return
    if depth != 0:
        result.add(LatexIssue(
            type="syntax",
            description=f"Unbalanced braces: {depth} unclosed '{{'",
            severity="error",
        ))


_ENV_BEGIN = re.compile(r"\\begin\{(\w+)\}")
_ENV_END = re.compile(r"\\end\{(\w+)\}")


def _check_environment_matching(content: str, result: LatexValidationResult) -> None:
    """Check that every \\begin{X} has a matching \\end{X} in order."""
    stack: list[str] = []

    for match in re.finditer(r"\\(begin|end)\{(\w+)\}", content):
        cmd, env = match.group(1), match.group(2)
        if cmd == "begin":
            stack.append(env)
        else:  # end
            if not stack:
                result.add(LatexIssue(
                    type="syntax",
                    description=f"\\end{{{env}}} without matching \\begin{{{env}}}",
                    severity="error",
                ))
            elif stack[-1] != env:
                result.add(LatexIssue(
                    type="syntax",
                    description=f"Environment mismatch: expected \\end{{{stack[-1]}}}, got \\end{{{env}}}",
                    severity="error",
                ))
                stack.pop()
            else:
                stack.pop()

    for env in stack:
        result.add(LatexIssue(
            type="syntax",
            description=f"\\begin{{{env}}} without matching \\end{{{env}}}",
            severity="error",
        ))
