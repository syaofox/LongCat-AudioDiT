import re
import librosa
import torch
import cn2an
from semantic_text_splitter import TextSplitter

def load_audio(wavpath, sr):
    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'["“”‘’]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def approx_duration_from_text(text, max_duration=30.0):
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)


# Common English abbreviations to expand
_ABBREVIATIONS = {
    "dr.": "doctor", "mr.": "mister", "mrs.": "missus", "ms.": "miss",
    "st.": "saint", "ave.": "avenue", "rd.": "road", "blvd.": "boulevard",
    "jr.": "junior", "sr.": "senior", "vs.": "versus", "etc.": "etcetera",
    "e.g.": "for example", "i.e.": "that is", "p.s.": "postscript",
    "a.m.": "a m", "p.m.": "p m",
}

_TERMINAL_PUNCT = set("。！？.!?")
_SOFT_PUNCT = set("，,、；;：:")
_ALL_PUNCT = _TERMINAL_PUNCT | _SOFT_PUNCT

_CONNECTOR_WORDS = ("但是", "然而", "不过", "可是", "而且", "并且", "同时", "另外", "然后", "接着", "但", "而", "且")


def normalize_mixed_text(text: str) -> str:
    """Normalize mixed Chinese-English text: numbers to Chinese, abbreviations expanded, then basic cleanup.

    Args:
        text: Raw text that may contain Arabic numerals and English abbreviations.

    Returns:
        Normalized text suitable for TTS tokenization.
    """
    # Expand common English abbreviations (case-insensitive)
    lower_text = text.lower()
    for abbr, expansion in _ABBREVIATIONS.items():
        lower_text = lower_text.replace(abbr, expansion)

    # Convert Arabic numerals to Chinese reading form
    # cn2an.transform with "an2cn" converts numbers in sentences to Chinese
    try:
        lower_text = cn2an.transform(lower_text, "an2cn")
    except Exception:
        pass  # Fallback: keep original text if conversion fails

    # Apply original normalization
    return normalize_text(lower_text)


def ensure_punctuation(text: str) -> str:
    """Ensure the text ends with a punctuation mark to force TTS pause.

    If the text already ends with terminal or soft punctuation, return as-is.
    Otherwise, append a comma (if text ends with connector words) or a period.

    Args:
        text: Text segment that may or may not end with punctuation.

    Returns:
        Text guaranteed to end with a punctuation mark.
    """
    text = text.rstrip()
    if not text:
        return text

    if text[-1] in _ALL_PUNCT:
        return text

    # Check if text ends with connector words that suggest continuation
    for connector in _CONNECTOR_WORDS:
        if text.endswith(connector):
            return text + "，"

    # Default: add a period
    return text + "。"


def split_text_semantic(text: str, max_chars: int = 100) -> list[str]:
    """Split text into semantic chunks with punctuation repair.

    Uses semantic-text-splitter to split text at natural boundaries
    (sentence/paragraph breaks) while respecting a maximum character limit.
    Each chunk is then passed through ensure_punctuation().

    Args:
        text: Text to split (should be a single paragraph, no empty lines).
        max_chars: Maximum characters per chunk. Defaults to 100
            (approximately 15-20 seconds of speech).

    Returns:
        List of text chunks, each ending with punctuation.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If text is short enough and already has proper punctuation, no need to split
    if len(text) <= max_chars:
        return [ensure_punctuation(text)]

    # Use semantic splitter with character-based length calculation
    splitter = TextSplitter(max_chars)
    chunks = splitter.chunks(text)

    # Clean up and ensure punctuation on each chunk
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            result.append(ensure_punctuation(chunk))

    return result if result else [ensure_punctuation(text)]