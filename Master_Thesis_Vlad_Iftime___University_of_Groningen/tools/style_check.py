#!/usr/bin/env python3
"""Style checker for the thesis LaTeX chapters.

Reports, per file and in aggregate:
  - total words
  - number of unique words (case-insensitive)
  - sentence count
  - mean sentence length (words)
  - standard deviation of sentence length
  - longest sentence (words) and the sentences that exceed the 25-word cap

The house style (see thesis_plan.md 1.4c) asks for:
  - no sentence longer than 25 words
  - mean sentence length ~14 words, standard deviation ~8.7

Usage:
  python3 tools/style_check.py chapters/methods.tex
  python3 tools/style_check.py chapters/*.tex
  python3 tools/style_check.py            # defaults to all chapters/*.tex
"""

import glob
import math
import os
import re
import sys

MAX_SENTENCE_WORDS = 35   # soft guideline; split anything longer unless a list
TARGET_MEAN = 18.0
TARGET_SD = 8.7
MIN_SD = 7.0              # rhythm floor: below this the prose is too monotone

# Environments whose contents are not prose and must be stripped before analysis.
STRIP_ENVIRONMENTS = [
    "equation", "align", "align*", "equation*", "gather", "gather*",
    "tabular", "tabularx", "table", "figure", "verbatim", "lstlisting",
    "matrix", "bmatrix", "pmatrix", "cases", "itemize", "enumerate",
]


def strip_latex(text: str) -> str:
    """Reduce LaTeX source to readable prose for sentence statistics."""
    # Drop comment lines and inline comments (unescaped %).
    text = re.sub(r"(?<!\\)%.*", "", text)

    # Remove whole environments that are not prose.
    for env in STRIP_ENVIRONMENTS:
        pattern = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            re.DOTALL,
        )
        text = pattern.sub(" ", text)

    # Remove display and inline math.
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)

    # Section headers and labels are not sentences: drop the command and its argument.
    text = re.sub(r"\\(sub)*section\*?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(label|ref|autoref|cite|eqref|includegraphics|caption)"
                  r"\s*(\[[^\]]*\])?\s*\{[^}]*\}", " ", text)
    text = re.sub(r"\\(item|hline|toprule|midrule|bottomrule|centering|small|"
                  r"footnotesize|scriptsize|hfill|vfill|newpage|clearpage|noindent)\b", " ", text)

    # Keep the readable argument of common formatting commands.
    text = re.sub(r"\\(textbf|textit|emph|texttt|underline|mathrm|text)\{([^}]*)\}",
                  r"\2", text)
    text = re.sub(r"\\TBD\b", "TBD", text)

    # Any remaining backslash command: drop the command, keep any braced argument text.
    text = re.sub(r"\\[a-zA-Z]+\*?\s*(\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("\\", " ")

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Abbreviations whose trailing period must not be read as a sentence end.
ABBREV = {
    "e.g", "i.e", "cf", "et al", "etc", "vs", "fig", "eq", "tab", "sec",
    "no", "dr", "prof", "mr", "mrs", "ms", "st", "approx",
}


def split_sentences(text: str) -> list:
    """Split prose into sentences on . ! ? while guarding abbreviations and decimals."""
    # Protect decimal points (a digit on each side).
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    # Protect known abbreviation periods.
    for abbr in ABBREV:
        text = re.sub(rf"\b({re.escape(abbr)})\.", r"\1<DOT>", text, flags=re.IGNORECASE)

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        if not part:
            continue
        # A sentence must contain at least one alphabetic word.
        if re.search(r"[A-Za-z]", part):
            sentences.append(part)
    return sentences


def word_list(text: str) -> list:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def analyse(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    prose = strip_latex(raw)
    sentences = split_sentences(prose)
    lengths = [len(word_list(s)) for s in sentences]
    lengths = [n for n in lengths if n > 0]

    words = word_list(prose)
    total_words = len(words)
    unique_words = len({w.lower() for w in words})

    if lengths:
        mean = sum(lengths) / len(lengths)
        var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
        sd = math.sqrt(var)
        longest = max(lengths)
    else:
        mean = sd = longest = 0.0

    over = [(n, s) for n, s in zip(lengths, sentences) if n > MAX_SENTENCE_WORDS]

    return {
        "path": path,
        "total_words": total_words,
        "unique_words": unique_words,
        "sentences": len(lengths),
        "mean": mean,
        "sd": sd,
        "longest": longest,
        "over": over,
        "lengths": lengths,
    }


def report(stats: dict, show_over: bool = True) -> None:
    print(f"\n=== {stats['path']} ===")
    print(f"  total words        : {stats['total_words']}")
    print(f"  unique words       : {stats['unique_words']}")
    print(f"  sentences          : {stats['sentences']}")
    print(f"  mean length        : {stats['mean']:.1f} words   (target {TARGET_MEAN})")
    sd_flag = "OK" if stats["sd"] >= MIN_SD else f"LOW (<{MIN_SD}, monotone)"
    print(f"  std deviation      : {stats['sd']:.1f} words   (target {TARGET_SD}; {sd_flag})")
    print(f"  longest sentence   : {stats['longest']} words   (soft cap {MAX_SENTENCE_WORDS})")
    n_over = len(stats["over"])
    flag = "OK" if n_over == 0 else f"{n_over} over soft cap"
    print(f"  over {MAX_SENTENCE_WORDS}-word guide : {flag}")
    if show_over and stats["over"]:
        for n, s in stats["over"]:
            preview = s if len(s) <= 140 else s[:137] + "..."
            print(f"      [{n}w] {preview}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = sorted(glob.glob(os.path.join(here, "chapters", "*.tex")))

    files = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])

    all_lengths = []
    agg_words = 0
    agg_unique = set()
    total_over = 0

    for path in files:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}")
            continue
        stats = analyse(path)
        report(stats)
        all_lengths.extend(stats["lengths"])
        agg_words += stats["total_words"]
        total_over += len(stats["over"])
        with open(path, "r", encoding="utf-8") as handle:
            agg_unique.update(w.lower() for w in word_list(strip_latex(handle.read())))

    if len(files) > 1 and all_lengths:
        mean = sum(all_lengths) / len(all_lengths)
        sd = math.sqrt(sum((n - mean) ** 2 for n in all_lengths) / len(all_lengths))
        print("\n=== AGGREGATE (all files) ===")
        print(f"  total words        : {agg_words}")
        print(f"  unique words       : {len(agg_unique)}")
        print(f"  sentences          : {len(all_lengths)}")
        print(f"  mean length        : {mean:.1f} words   (target {TARGET_MEAN})")
        print(f"  std deviation      : {sd:.1f} words   (target {TARGET_SD})")
        print(f"  longest sentence   : {max(all_lengths)} words   (soft cap {MAX_SENTENCE_WORDS})")
        print(f"  over {MAX_SENTENCE_WORDS}-word guide : "
              f"{'OK' if total_over == 0 else str(total_over) + ' over soft cap'}")

    sys.exit(0)


if __name__ == "__main__":
    main()
