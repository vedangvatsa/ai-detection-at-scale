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
HIDDEN_CHARS_RE = re.compile(r'[\u200b-\u200d\ufeff\u200e\u200f\u202a-\u202e\xad]')

def normalize_text_defensive(text: str) -> str:
    """
    Cleans text against adversarial character-level attacks:
    - Normalizes Unicode representations to NFC.
    - Strips zero-width characters and hidden formatting.
    - Replaces homoglyphs (confusable lookalike letters) with Latin equivalents.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Strip hidden zero-width / control characters
    text = HIDDEN_CHARS_RE.sub('', text)
    
    # 3. Swap lookalike homoglyphs
    chars = []
    for char in text:
        chars.append(HOMOGLYPH_MAP.get(char, char))
        
    cleaned_text = "".join(chars)
    
    # 4. Collapse extra whitespace to normal spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text
