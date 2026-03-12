"""
Tokenizers for text and LaTeX output sequences.

Character-level for text, token-level for LaTeX.
"""

from __future__ import annotations

from typing import List, Dict


class TextTokenizer:
    """Character-level tokenizer for handwritten text recognition.

    Vocabulary: special tokens + ASCII printable + extended Latin + common symbols.
    Total: ~150 tokens.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.special_tokens = {
            "<PAD>": self.PAD,
            "<BOS>": self.BOS,
            "<EOS>": self.EOS,
            "<UNK>": self.UNK,
        }

        # ASCII printable characters (32-126)
        ascii_chars = [chr(i) for i in range(32, 127)]

        # Extended Latin characters common in European languages
        extended = list("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
                        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
                        "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
                        "šžŠŽ")

        # Common math/science symbols that appear in text
        symbols = list("°±²³µ·¼½¾×÷€£¥©®™∞≈≠≤≥")

        all_chars = ascii_chars + extended + symbols
        # Deduplicate while preserving order
        seen = set()
        unique_chars = []
        for c in all_chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.id_to_char[idx] = token

        # Add characters starting after special tokens
        offset = len(self.special_tokens)
        for i, char in enumerate(unique_chars):
            idx = offset + i
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

        self._vocab_size = offset + len(unique_chars)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode a text string to token IDs."""
        ids = []
        if add_special:
            ids.append(self.BOS)
        for char in text:
            ids.append(self.char_to_id.get(char, self.UNK))
        if add_special:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decode token IDs to text string."""
        chars = []
        for idx in ids:
            if strip_special and idx in (self.PAD, self.BOS, self.EOS):
                continue
            token = self.id_to_char.get(idx, "")
            if not token.startswith("<"):
                chars.append(token)
        return "".join(chars)

    def pad_sequence(self, ids: List[int], max_length: int) -> List[int]:
        """Pad or truncate to max_length."""
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [self.PAD] * (max_length - len(ids))


class MathTokenizer:
    """Token-level tokenizer for LaTeX math expressions.

    Vocabulary: special tokens + single characters + LaTeX commands.
    Total: ~500 tokens.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.special_tokens = {
            "<PAD>": self.PAD,
            "<BOS>": self.BOS,
            "<EOS>": self.EOS,
            "<UNK>": self.UNK,
        }

        # Single characters used in math
        single_chars = list(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "+-=<>()[]{}|/\\.,;:!?'\"_ ^~`@#$%&*"
        )

        # LaTeX commands (sorted for consistency)
        latex_commands = sorted([
            # Greek lowercase
            "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
            "\\zeta", "\\eta", "\\theta", "\\vartheta", "\\iota", "\\kappa",
            "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma",
            "\\tau", "\\upsilon", "\\phi", "\\varphi", "\\chi", "\\psi", "\\omega",
            # Greek uppercase
            "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi",
            "\\Sigma", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega",
            # Operators
            "\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", "\\oint",
            "\\iint", "\\iiint", "\\lim", "\\sup", "\\inf", "\\min", "\\max",
            "\\arg", "\\det", "\\dim", "\\gcd", "\\ker", "\\deg",
            # Functions
            "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc",
            "\\arcsin", "\\arccos", "\\arctan",
            "\\sinh", "\\cosh", "\\tanh", "\\coth",
            "\\log", "\\ln", "\\exp", "\\lg",
            # Relations
            "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\sim",
            "\\simeq", "\\cong", "\\propto", "\\ll", "\\gg",
            "\\subset", "\\supset", "\\subseteq", "\\supseteq",
            "\\in", "\\notin", "\\ni",
            "\\prec", "\\succ", "\\preceq", "\\succeq",
            # Arrows
            "\\rightarrow", "\\leftarrow", "\\leftrightarrow",
            "\\Rightarrow", "\\Leftarrow", "\\Leftrightarrow",
            "\\mapsto", "\\to", "\\gets",
            "\\uparrow", "\\downarrow", "\\updownarrow",
            "\\nearrow", "\\searrow", "\\nwarrow", "\\swarrow",
            "\\longrightarrow", "\\longleftarrow", "\\longmapsto",
            # Binary operators
            "\\pm", "\\mp", "\\times", "\\div", "\\cdot", "\\circ",
            "\\bullet", "\\star", "\\ast", "\\oplus", "\\otimes",
            "\\odot", "\\wedge", "\\vee", "\\cap", "\\cup",
            "\\setminus", "\\triangle", "\\nabla",
            # Accents / decorations
            "\\hat", "\\bar", "\\vec", "\\dot", "\\ddot", "\\tilde",
            "\\overline", "\\underline", "\\overbrace", "\\underbrace",
            "\\overrightarrow", "\\overleftarrow", "\\widehat", "\\widetilde",
            # Delimiters
            "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg",
            "\\lfloor", "\\rfloor", "\\lceil", "\\rceil",
            "\\langle", "\\rangle", "\\lvert", "\\rvert",
            "\\lVert", "\\rVert",
            # Dots
            "\\dots", "\\cdots", "\\vdots", "\\ddots", "\\ldots",
            # Fonts
            "\\mathbb", "\\mathbf", "\\mathrm", "\\mathcal", "\\mathfrak",
            "\\mathsf", "\\mathit", "\\text", "\\textbf", "\\textit",
            # Logic
            "\\forall", "\\exists", "\\nexists", "\\neg", "\\land", "\\lor",
            "\\implies", "\\iff", "\\therefore", "\\because",
            # Misc symbols
            "\\infty", "\\partial", "\\emptyset", "\\varnothing",
            "\\angle", "\\perp", "\\parallel", "\\mid", "\\nmid",
            "\\prime", "\\backprime", "\\hbar", "\\ell",
            "\\Re", "\\Im", "\\wp", "\\aleph",
            # Spacing
            "\\quad", "\\qquad", "\\,", "\\;", "\\!", "\\ ",
            # Environments
            "\\begin", "\\end",
            "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
            "cases", "align", "aligned", "array", "gather",
            "equation", "split",
            # Line break
            "\\\\", "\\newline", "\\hline",
            # Alignment
            "&",
            # Other structural
            "\\displaystyle", "\\textstyle", "\\scriptstyle",
            "\\limits", "\\nolimits",
            "\\stackrel", "\\overset", "\\underset",
            "\\binom", "\\tbinom", "\\dbinom",
            "\\choose",
        ])

        # Build vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        offset = len(self.special_tokens)

        # Single characters
        for i, char in enumerate(single_chars):
            idx = offset + i
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
        offset += len(single_chars)

        # LaTeX commands (deduplicated)
        seen = set(single_chars)
        for cmd in latex_commands:
            if cmd not in seen:
                seen.add(cmd)
                self.token_to_id[cmd] = offset
                self.id_to_token[offset] = cmd
                offset += 1

        self._vocab_size = offset

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenize(self, latex: str) -> List[str]:
        """Tokenize a LaTeX string into tokens."""
        tokens = []
        i = 0
        while i < len(latex):
            # Skip whitespace
            if latex[i] == ' ':
                i += 1
                continue

            # LaTeX command (starts with \)
            if latex[i] == '\\':
                # Check for \\ (line break)
                if i + 1 < len(latex) and latex[i + 1] == '\\':
                    tokens.append("\\\\")
                    i += 2
                    continue

                # Collect command name
                j = i + 1
                if j < len(latex) and not latex[j].isalpha():
                    # Single-char commands like \, \; \! \{
                    cmd = latex[i:j + 1]
                    tokens.append(cmd)
                    i = j + 1
                else:
                    while j < len(latex) and latex[j].isalpha():
                        j += 1
                    cmd = latex[i:j]
                    tokens.append(cmd)
                    i = j
            else:
                # Single character
                tokens.append(latex[i])
                i += 1

        return tokens

    def encode(self, latex: str, add_special: bool = True) -> List[int]:
        """Encode a LaTeX string to token IDs."""
        tokens = self.tokenize(latex)
        ids = []
        if add_special:
            ids.append(self.BOS)
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.UNK))
        if add_special:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decode token IDs back to LaTeX string."""
        tokens = []
        for idx in ids:
            if strip_special and idx in (self.PAD, self.BOS, self.EOS):
                continue
            token = self.id_to_token.get(idx, "")
            if not token.startswith("<"):
                tokens.append(token)
        # Join with appropriate spacing
        result = []
        for i, token in enumerate(tokens):
            if token.startswith("\\") and token not in ("\\\\", "\\,", "\\;", "\\!", "\\ "):
                # Add space before LaTeX commands if previous token was also alpha
                if result and result[-1][-1:].isalpha():
                    result.append(" ")
            result.append(token)
        return "".join(result)

    def pad_sequence(self, ids: List[int], max_length: int) -> List[int]:
        """Pad or truncate to max_length."""
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [self.PAD] * (max_length - len(ids))
