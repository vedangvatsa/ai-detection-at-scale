#!/usr/bin/env python3
"""
Multilingual word lists for stylometric AI text detection.

Provides translated hedge, booster, connector, and self-mention word lists
for the top 5 languages: English, Spanish, French, German, and Chinese.

Usage:
    from tool.multilingual import get_word_lists, detect_language
    word_lists = get_word_lists('es')
"""
import re
import unicodedata


# ── English (baseline, from feature_extractor.py) ──────────────────────────

EN = {
    'hedge': {
        'may', 'might', 'could', 'possibly', 'perhaps', 'probably', 'likely',
        'appears', 'seems', 'suggests', 'indicates', 'tentatively', 'presumably',
        'arguably', 'approximately', 'generally', 'often', 'sometimes', 'tend',
        'tends', 'tended', 'appear', 'suggest', 'indicate', 'assume', 'assumes',
        'assumed', 'possible', 'potential', 'potentially', 'conceivably',
        'loosely', 'roughly',
    },
    'booster': {
        'clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely',
        'demonstrates', 'demonstrate', 'proves', 'prove', 'establishes',
        'establish', 'confirms', 'confirm', 'shows', 'always', 'never',
        'absolutely', 'conclusively', 'evidently', 'indeed', 'strongly',
        'unambiguously', 'fundamentally', 'necessarily', 'undeniably',
    },
    'connector': {
        'however', 'therefore', 'thus', 'hence', 'consequently', 'nevertheless',
        'furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise',
        'similarly', 'conversely', 'alternatively', 'subsequently', 'finally',
        'first', 'second', 'third', 'meanwhile', 'nonetheless', 'instead',
        'otherwise', 'accordingly',
    },
    'self_mention': {'we', 'our', 'us', 'i', 'my', 'me', 'myself', 'ourselves'},
}

# ── Spanish ────────────────────────────────────────────────────────────────

ES = {
    'hedge': {
        'puede', 'podria', 'posiblemente', 'quizas', 'tal', 'vez', 'probablemente',
        'parece', 'sugiere', 'indica', 'aproximadamente', 'generalmente',
        'a menudo', 'a veces', 'tiende', 'posible', 'potencial', 'presumiblemente',
        'arguablemente', 'concebiblemente',
    },
    'booster': {
        'claramente', 'obviamente', 'indudablemente', 'ciertamente', 'definitivamente',
        'demuestra', 'demuestran', 'prueba', 'prueban', 'confirma', 'confirman',
        'muestra', 'muestran', 'siempre', 'nunca', 'absolutamente', 'evidentemente',
        'efectivamente', 'fuertemente',
    },
    'connector': {
        'sin', 'embargo', 'por', 'lo', 'tanto', 'asi', 'por', 'consiguiente',
        'ademas', 'tambien', 'igualmente', 'de', 'manera', 'similar',
        'por', 'el', 'contrario', 'finalmente', 'mientras', 'tanto',
        'no', 'obstante', 'en', 'cambio', 'de', 'otra', 'manera',
    },
    'self_mention': {'nosotros', 'nuestro', 'nos', 'yo', 'mi', 'me', 'me mismo'},
}

# ── French ─────────────────────────────────────────────────────────────────

FR = {
    'hedge': {
        'peut', 'pourrait', 'possiblement', 'peut-etre', 'probablement',
        'semble', 'suggere', 'indique', 'approximativement', 'generalement',
        'souvent', 'parfois', 'tend', 'possible', 'potentiel', 'potentiellement',
        'presumablement', 'arguablement',
    },
    'booster': {
        'clairement', 'evidemment', 'indeniablement', 'certainement', 'definitivement',
        'demontre', 'demontrent', 'prouve', 'prouvent', 'confirme', 'confirment',
        'montre', 'montrent', 'toujours', 'jamais', 'absolument', 'evidemment',
        'effectivement', 'fortement',
    },
    'connector': {
        'cependant', 'toutefois', 'donc', 'par', 'consequent', 'ainsi',
        'de', 'plus', 'en', 'outre', 'egalement', 'de', 'maniere', 'similaire',
        'en', 'revanche', 'finalement', 'pendant', 'ce', 'temps',
        'neanmoins', 'en', 'revanche', 'autrement',
    },
    'self_mention': {'nous', 'notre', 'je', 'mon', 'ma', 'me', 'moi', 'moi-meme'},
}

# ── German ─────────────────────────────────────────────────────────────────

DE = {
    'hedge': {
        'kann', 'koennte', 'moeglicherweise', 'vielleicht', 'wahrscheinlich',
        'scheint', 'deutet', 'hin', 'weist', 'etwa', 'im', 'allgemeinen',
        'oft', 'manchmal', 'tendiert', 'moeglich', 'potenziell',
        'vermutlich', 'argumentierbar',
    },
    'booster': {
        'klar', 'offensichtlich', 'zweifellos', 'gewiss', 'definitiv',
        'demonstriert', 'beweist', 'bestaetigt', 'zeigt', 'immer', 'nie',
        'absolut', 'eindeutig', 'tatsaechlich', 'stark', 'notwendigerweise',
    },
    'connector': {
        'jedoch', 'daher', 'folglich', 'somit', 'ausserdem', 'zudem',
        'ebenfalls', 'aehnlich', 'umgekehrt', 'schliesslich', 'waehrend',
        'nichtsdestotrotz', 'stattdessen', 'andernfalls', 'entsprechend',
    },
    'self_mention': {'wir', 'unser', 'uns', 'ich', 'mein', 'mich', 'mir', 'mich selbst'},
}

# ── Chinese ────────────────────────────────────────────────────────────────

ZH = {
    'hedge': {
        '可能', '也许', '或许', '大概', '似乎', '表明', '指出', '大约',
        '一般', '通常', '往往', '有时', '倾向于', '可能的', '潜在的',
    },
    'booster': {
        '显然', '明显地', '无疑', '当然', '确实', '证明', '证实',
        '显示', '总是', '从不', '绝对', '明确地', '事实上', '强烈',
    },
    'connector': {
        '然而', '因此', '从而', '此外', '而且', '同样', '类似地',
        '相反', '最后', '同时', '尽管如此', '反而', '否则', '相应地',
    },
    'self_mention': {'我们', '我们的', '我', '我的', '自己'},
}

# ── Language registry ─────────────────────────────────────────────────────

WORD_LISTS = {
    'en': EN,
    'es': ES,
    'fr': FR,
    'de': DE,
    'zh': ZH,
}

LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
}


def detect_language(text, sample_size=500):
    """Heuristic language detection based on script and common words.

    Returns a language code: 'en', 'es', 'fr', 'de', 'zh', or 'unknown'.
    """
    sample = text[:sample_size]

    # Chinese: CJK characters
    cjk_count = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    if cjk_count > len(sample) * 0.1:
        return 'zh'

    # Latin script languages — use common function words
    lower = sample.lower()
    scores = {}

    markers = {
        'en': [' the ', ' is ', ' are ', ' and ', ' of ', ' to ', ' in '],
        'es': [' el ', ' la ', ' es ', ' y ', ' de ', ' en ', ' que '],
        'fr': [' le ', ' la ', ' est ', ' et ', ' de ', ' en ', ' que '],
        'de': [' der ', ' die ', ' das ', ' und ', ' von ', ' in ', ' ist '],
    }

    for lang, words in markers.items():
        scores[lang] = sum(lower.count(w) for w in words)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return 'unknown'


def get_word_lists(lang='en'):
    """Get word lists for a given language.

    Returns a dict with keys: 'hedge', 'booster', 'connector', 'self_mention'.
    Falls back to English if language not available.
    """
    return WORD_LISTS.get(lang, EN)


def available_languages():
    """Return list of supported language codes."""
    return list(WORD_LISTS.keys())
