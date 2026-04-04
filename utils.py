import re
import json
import os
import librosa
import torch
import numpy as np
import pyloudnorm as pyln
import cn2an
from semantic_text_splitter import TextSplitter

# Polyphone rules config file path (relative to this file or absolute)
_POLYPHONE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyphone_rules.json")

# Default rules used when config file is missing or invalid
_DEFAULT_POLYPHONE_RULES = {
    "银行[行]长": "航",
}

def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    """Normalize audio loudness to target LUFS using EBU R128 (K-weighting).

    Args:
        audio: Mono audio array (float32).
        sr: Sample rate.
        target_lufs: Target loudness in LUFS. Default -18.0 for TTS reference audio.

    Returns:
        Loudness-normalized audio array.
    """
    if audio.size == 0:
        return audio

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)

    if np.isinf(loudness) or np.isnan(loudness):
        return audio

    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    normalized = audio * gain_linear

    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)

    return normalized.astype(np.float32)


def load_audio(wavpath, sr):
    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    audio = normalize_loudness(audio, sr)
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


def normalize_mixed_text(text: str, country: str = "auto") -> str:
    """Normalize mixed Chinese-English text: numbers to Chinese, abbreviations expanded, then basic cleanup.

    Args:
        text: Raw text that may contain Arabic numerals and English abbreviations.
        country: Target language/country for normalization.
            "auto" - auto-detect based on text content
            "zh" - Chinese (convert numbers to Chinese reading form)
            "en" - English (keep numbers as Arabic numerals, expand abbreviations)

    Returns:
        Normalized text suitable for TTS tokenization.
    """
    # Expand common English abbreviations (case-insensitive)
    lower_text = text.lower()
    for abbr, expansion in _ABBREVIATIONS.items():
        lower_text = lower_text.replace(abbr, expansion)

    # Determine if we should convert numbers to Chinese
    should_convert_to_cn = country == "zh"
    if country == "auto":
        # Auto-detect: if text contains more Chinese characters than English letters, convert
        num_zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        num_en = sum(1 for c in text if c.isascii() and c.isalpha())
        should_convert_to_cn = num_zh >= num_en

    # Convert Arabic numerals to Chinese reading form (Chinese mode only)
    if should_convert_to_cn:
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
        chunk = ensure_punctuation(text)
        print(f"  [语义分割] 文本较短({len(text)}字), 不分割: \"{chunk}\"")
        return [chunk]

    # Use semantic splitter with character-based length calculation
    splitter = TextSplitter(max_chars)
    raw_chunks = splitter.chunks(text)

    # Clean up and ensure punctuation on each chunk
    result = []
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if chunk:
            fixed = ensure_punctuation(chunk)
            punct_added = fixed != chunk.rstrip()
            print(f"  [语义分割] 段落 {i+1}/{len(raw_chunks)} ({len(chunk)}字) [补标点: {'是' if punct_added else '否'}]: \"{fixed}\"")
            result.append(fixed)

    if not result:
        fixed = ensure_punctuation(text)
        print(f"  [语义分割] 分割结果为空, 使用原文: \"{fixed}\"")
        result = [fixed]

    return result


# ─── Polyphone Rules ────────────────────────────────────────────────────────

def load_polyphone_rules(config_path: str | None = None) -> dict[str, str]:
    """Load polyphone replacement rules from a JSON config file.

    Args:
        config_path: Path to the JSON config file. Defaults to polyphone_rules.json
            in the same directory as this module.

    Returns:
        Dict of pattern → replacement rules. Falls back to defaults if file is missing.
    """
    path = config_path or _POLYPHONE_CONFIG_PATH
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", {})
            if rules:
                print(f"[多音字] 从 {path} 加载了 {len(rules)} 条规则")
                return rules
        except Exception as e:
            print(f"[多音字] 加载配置文件失败 ({e}), 使用默认规则")
    print(f"[多音字] 使用内置默认规则 ({len(_DEFAULT_POLYPHONE_RULES)} 条)")
    return dict(_DEFAULT_POLYPHONE_RULES)


def save_polyphone_rules(rules: dict[str, str], config_path: str | None = None) -> str:
    """Save polyphone replacement rules to a JSON config file.

    Args:
        rules: Dict of pattern → replacement rules.
        config_path: Path to save the JSON config file. Defaults to polyphone_rules.json
            in the same directory as this module.

    Returns:
        Path to the saved file.
    """
    path = config_path or _POLYPHONE_CONFIG_PATH
    data = {"rules": rules}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[多音字] 已保存 {len(rules)} 条规则到 {path}")
    return path


def apply_polyphone_rules(text: str, rules: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Apply polyphone replacement rules to text.

    Rule format:
        - "[行长]" → "航长" : No context, replace the whole pattern (brackets stripped)
        - "银行[行]长" → "航" : With context, replace only the bracketed char in the full pattern

    Rules are applied in descending order of pattern length to prioritize
    context-specific rules over generic ones.

    Args:
        text: Input text to process.
        rules: Dict of pattern → replacement. If None, loads from config file.

    Returns:
        Tuple of (processed_text, list_of_log_messages).
    """
    if rules is None:
        rules = load_polyphone_rules()

    if not rules:
        return text, []

    logs = []

    # Sort rules by pattern length descending (longer patterns = more context = higher priority)
    sorted_rules = sorted(rules.items(), key=lambda x: len(x[0]), reverse=True)

    result = text

    for pattern, replacement in sorted_rules:
        if "[" not in pattern:
            # No context: pattern is "[词]" → replace "词" with replacement
            # e.g., "[行长]" → "航长" means replace "行长" with "航长"
            word = pattern.strip("[]")
            if word not in result:
                continue
            # Check if the text has the bracketed form "[词]" — preserve brackets
            bracketed_form = f"[{word}]"
            if bracketed_form in result:
                result = result.replace(bracketed_form, f"[{replacement}]")
                logs.append(f"[多音字] \"{bracketed_form}\" → \"[{replacement}]\" (规则: {pattern}→{replacement})")
            else:
                result = result.replace(word, replacement)
                logs.append(f"[多音字] \"{word}\" → \"{replacement}\" (规则: {pattern}→{replacement})")
        else:
            # With context: pattern has brackets inside
            # e.g., "银行[行]长" → "航" means find "银行行长", replace "行" with "航"
            bracket_start = pattern.index("[")
            bracket_end = pattern.index("]")
            context_before = pattern[:bracket_start]
            char_to_replace = pattern[bracket_start + 1:bracket_end]
            context_after = pattern[bracket_end + 1:]

            full_pattern = context_before + char_to_replace + context_after
            full_replacement = context_before + replacement + context_after

            if full_pattern in result:
                result = result.replace(full_pattern, full_replacement)
                logs.append(f"[多音字] \"{full_pattern}\" → \"{full_replacement}\" (规则: {pattern}→{replacement})")

    return result, logs