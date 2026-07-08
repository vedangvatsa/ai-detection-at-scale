import unicodedata
import re

# Mapping of common Unicode Cyrillic/Greek homoglyphs to standard Latin lookalikes
HOMOGLYPH_MAP = {
    # Lowercase
    'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y',
    'і': 'i', 'ѕ': 's', 'ј': 'j', 'ԁ': 'd', 'һ': 'h', 'ԝ': 'w',
    # Uppercase
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I', 'Ј': 'J',
    'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Ѕ': 'S', 'Т': 'T', 'Х': 'X',
    'Ү': 'Y'
}

# Regex to catch zero-width spaces and other hidden formatting/obfuscation characters
HIDDEN_CHARS_RE = re.compile(
    r'[\u200b-\u200d\ufeff\u200e\u200f\u202a-\u202e\xad'
    r'\u2060-\u206f\u00ad\u0e00\u115f\u1160\u3164\uffa0'
    r'\u180e\u2000-\u200a\u2028\u2029]'
)

# Whitespace attacks: repeated spaces, tabs, newlines, zero-width joiners
_WHITESPACE_ATTACK_RE = re.compile(r'[ \t\r\n\u200b\u200c\u200d\u2060]{2,}')

# Punctuation noise: repeated non-word characters used to break stylometric features
_PUNCT_NOISE_RE = re.compile(r'([^\w\s])\1{2,}')

def normalize_text_defensive(text: str) -> str:
    """
    Cleans text against adversarial character-level attacks:
    - Normalizes Unicode representations to NFC.
    - Strips zero-width characters, control characters, and invisible formatting.
    - Replaces homoglyphs (confusable lookalike letters) with Latin equivalents.
    - Collapses whitespace/punctuation attacks that evade stylometric features.

    Note: This is a preprocessing defense against *character-level* adversarial
    text. It does not protect against semantic paraphrasing, prompt-injection,
    synonym substitution, or back-translation.
    """
    if not isinstance(text, str):
        return ""

    # 1. NFC normalization
    text = unicodedata.normalize('NFC', text)

    # 2. Strip hidden zero-width / control characters
    text = HIDDEN_CHARS_RE.sub('', text)

    # 3. Swap lookalike homoglyphs
    cleaned_text = ''.join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)

    # 4. Collapse whitespace attacks (zero-width joiners already removed)
    cleaned_text = _WHITESPACE_ATTACK_RE.sub(' ', cleaned_text)

    # 5. Deduplicate repeated punctuation/noise while keeping sentence structure
    cleaned_text = _PUNCT_NOISE_RE.sub(r'\1', cleaned_text)

    # 6. Collapse extra whitespace to normal spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text
