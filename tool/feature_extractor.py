#!/usr/bin/env python3
"""
Shared feature extraction module — 31 stylometric features.
Used by the inference API, extended analysis scripts, and ensemble pipeline.

Original 11 features: MTLD, sentence CV, mean sentence length, self-mention density,
connector density, opener ratio, hedge density, booster density, char n-gram entropy,
word repetition rate, punctuation entropy.

Extended 20 features: Flesch-Kincaid grade, Flesch reading ease, Gunning fog index,
SMOG index, positive sentiment density, negative sentiment density, sentiment polarity,
exclamation density, question density, passive voice density, subordination density,
preposition density, adjective density, adverb density, nominalization density,
capitalized entity density, number density, acronym density, url_email density,
quote density.
"""
import re
import math
import unicodedata
from collections import Counter
import numpy as np


# ── Original word lists ────────────────────────────────────────────────────

HEDGE_WORDS = {
    'may', 'might', 'could', 'possibly', 'perhaps', 'probably', 'likely',
    'appears', 'seems', 'suggests', 'indicates', 'tentatively', 'presumably',
    'arguably', 'approximately', 'generally', 'often', 'sometimes', 'tend',
    'tends', 'tended', 'appear', 'suggest', 'indicate', 'assume', 'assumes',
    'assumed', 'appear to', 'seem to', 'likely to', 'possible', 'potential',
    'potentially', 'conceivably', 'presumably', 'loosely', 'roughly',
}

BOOSTER_WORDS = {
    'clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely',
    'demonstrates', 'demonstrate', 'proves', 'prove', 'establishes',
    'establish', 'confirms', 'confirm', 'shows', 'always', 'never',
    'absolutely', 'conclusively', 'evidently', 'indeed', 'strongly',
    'unambiguously', 'fundamentally', 'necessarily', 'undeniably',
}

CONNECTOR_WORDS = {
    'however', 'therefore', 'thus', 'hence', 'consequently', 'nevertheless',
    'furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise',
    'similarly', 'conversely', 'alternatively', 'subsequently', 'finally',
    'first', 'second', 'third', 'firstly', 'secondly', 'thirdly', 'lastly',
    'meanwhile', 'nonetheless', 'instead', 'otherwise', 'accordingly',
    'for example', 'for instance', 'in contrast', 'in addition', 'in conclusion',
    'in summary', 'as a result', 'on the other hand', 'on the contrary',
}

SELF_MENTION_WORDS = {'we', 'our', 'us', 'i', 'my', 'me', 'myself', 'ourselves'}

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
    'and', 'or', 'but', 'not', 'it', 'its', 'this', 'that',
    'these', 'those', 'as', 'if', 'so', 'than', 'then', 'when',
    'where', 'which', 'who', 'what', 'how', 'all', 'more', 'most',
}

# ── Extended word lists ────────────────────────────────────────────────────

POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'best', 'better', 'love', 'loved', 'perfect', 'perfectly', 'happy',
    'glad', 'pleased', 'delighted', 'satisfied', 'enjoy', 'enjoyed',
    'enjoyable', 'superb', 'outstanding', 'remarkable', 'brilliant',
    'beautiful', 'stunning', 'impressive', 'positive', 'beneficial',
    'successful', 'effective', 'efficient', 'valuable', 'important',
    'significant', 'notable', 'noteworthy', 'commendable', 'exceptional',
    'favorable', 'promising', 'optimistic', 'encouraging', 'rewarding',
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'worse', 'hate',
    'hated', 'disappointing', 'disappointed', 'disappointment', 'poor',
    'poorly', 'fail', 'failed', 'failure', 'wrong', 'incorrect', 'error',
    'error', 'flaw', 'flawed', 'defective', 'negative', 'harmful',
    'damaging', 'concerning', 'troubling', 'problematic', 'inadequate',
    'insufficient', 'ineffective', 'inefficient', 'useless', 'pointless',
    'boring', 'dull', 'uninteresting', 'confusing', 'confused', 'unclear',
    'vague', 'ambiguous', 'questionable', 'dubious', 'flawed', 'weak',
}

PREPOSITIONS = {
    'about', 'above', 'across', 'after', 'against', 'along', 'among',
    'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside',
    'between', 'beyond', 'by', 'down', 'during', 'except', 'for',
    'from', 'in', 'inside', 'into', 'near', 'of', 'off', 'on',
    'onto', 'out', 'outside', 'over', 'through', 'throughout', 'to',
    'toward', 'under', 'until', 'up', 'upon', 'with', 'within', 'without',
}

COMMON_ADJECTIVE_SUFFIXES = ('ful', 'less', 'ous', 'ive', 'able', 'ible',
                             'al', 'ant', 'ent', 'ic', 'ish', 'like')

COMMON_ADVERB_SUFFIXES = ('ly', 'wise', 'ward', 'wards')

NOMINALIZATION_SUFFIXES = ('tion', 'sion', 'ment', 'ness', 'ity',
                           'ance', 'ence', 'ship', 'hood', 'dom')

PASSIVE_INDICATORS = re.compile(
    r'\b(?:is|are|was|were|be|been|being|has|have|had)\s+'
    r'(?:\w+ed|made|done|seen|given|taken|known|shown|found|'
    r'held|kept|left|put|set|sent|brought|bought|caught|taught|'
    r'sold|told|paid|laid|said|read|written|driven|chosen|fallen|'
    r'eaten|given|hidden|ridden|risen|woven|stolen|forgotten)\b',
    re.IGNORECASE,
)

SUBORDINATING_CONJUNCTIONS = {
    'because', 'although', 'though', 'while', 'whereas', 'since',
    'unless', 'until', 'before', 'after', 'as', 'if', 'whether',
    'provided', 'supposing', 'assuming', 'given', 'even', 'once',
    'whenever', 'wherever', 'whoever', 'whichever', 'so', 'that',
}


# ── Tokenization ──────────────────────────────────────────────────────────

def tokenize_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]


def tokenize_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^aeiouy]es|ed|[^aeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    groups = re.findall(r'[aeiouy]+', word)
    return max(len(groups), 1)


# ── MTLD ──────────────────────────────────────────────────────────────────

def mtld_forward(words, threshold=0.72):
    if len(words) < 10:
        return 0.0
    factor_count = 0.0
    token_count = 0
    types = set()
    for i, w in enumerate(words):
        token_count += 1
        types.add(w)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factor_count += 1
            token_count = 0
            types = set()
    if token_count > 0:
        ttr = len(types) / token_count
        factor_count += (1.0 - ttr) / (1.0 - threshold)
    if factor_count == 0:
        return len(words)
    return len(words) / factor_count


def compute_mtld(words):
    if len(words) < 10:
        return np.nan
    forward = mtld_forward(words)
    backward = mtld_forward(list(reversed(words)))
    return (forward + backward) / 2.0


# ── Original 11 features ──────────────────────────────────────────────────

def _extract_original_features(text, words, sents, lang='en'):
    n_words = len(words)
    n_sents = len(sents)
    words_per_1000 = n_words / 1000.0

    sent_lengths = [len(tokenize_words(s)) for s in sents if tokenize_words(s)]
    if not sent_lengths:
        return None
    mean_sent_len = np.mean(sent_lengths)

    if len(sent_lengths) >= 2 and mean_sent_len > 0:
        sent_cv = np.std(sent_lengths, ddof=1) / mean_sent_len
    else:
        sent_cv = 0.0

    mtld_val = compute_mtld(words)
    mtld = 0.0 if np.isnan(mtld_val) else mtld_val

    if lang == 'en' or lang == 'unknown':
        hedge_words = HEDGE_WORDS
        booster_words = BOOSTER_WORDS
        connector_words = CONNECTOR_WORDS
        self_mention_words = SELF_MENTION_WORDS
    else:
        from tool.multilingual import get_word_lists
        wlist = get_word_lists(lang)
        hedge_words = wlist.get('hedge', HEDGE_WORDS)
        booster_words = wlist.get('booster', BOOSTER_WORDS)
        connector_words = wlist.get('connector', CONNECTOR_WORDS)
        self_mention_words = wlist.get('self_mention', SELF_MENTION_WORDS)

    self_count = sum(1 for w in words if w in self_mention_words)
    self_density = self_count / max(words_per_1000, 0.001)

    text_lower = text.lower()
    conn_count = 0
    for cw in connector_words:
        conn_count += len(re.findall(r'\b' + re.escape(cw) + r'\b', text_lower))
    conn_density = conn_count / max(words_per_1000, 0.001)

    opener_count = 0
    for s in sents:
        s_lower = s.lower().strip()
        for cw in connector_words:
            if s_lower.startswith(cw + ' ') or s_lower.startswith(cw + ','):
                opener_count += 1
                break
    opener_ratio = opener_count / max(n_sents, 1)

    hedge_count = sum(1 for w in words if w in hedge_words)
    for mw in ['appear to', 'seem to', 'likely to', 'for example', 'for instance']:
        hedge_count += len(re.findall(r'\b' + re.escape(mw) + r'\b', text_lower))
    hedge_density = hedge_count / max(words_per_1000, 0.001)

    boost_count = sum(1 for w in words if w in booster_words)
    boost_density = boost_count / max(words_per_1000, 0.001)

    chars = re.sub(r'\s+', ' ', text.lower())
    if len(chars) >= 3:
        trigrams = [chars[i:i+3] for i in range(len(chars)-2)]
        tg_counts = Counter(trigrams)
        total = sum(tg_counts.values())
        char_entropy = -sum((c/total) * math.log2(c/total) for c in tg_counts.values())
    else:
        char_entropy = 0.0

    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if content_words:
        wf = Counter(content_words)
        repeated = sum(1 for w in content_words if wf[w] > 1)
        rep_rate = repeated / len(content_words)
    else:
        rep_rate = 0.0

    puncts = [c for c in text if c in '.,;:!?()[]{}"\'-']
    if puncts:
        pf = Counter(puncts)
        total_p = len(puncts)
        punct_entropy = -sum((c/total_p) * math.log2(c/total_p) for c in pf.values())
    else:
        punct_entropy = 0.0

    return {
        'mean_sent_len': mean_sent_len,
        'sent_cv': sent_cv,
        'mtld': mtld,
        'self_mention_density': self_density,
        'connector_density': conn_density,
        'opener_ratio': opener_ratio,
        'hedge_density': hedge_density,
        'boost_density': boost_density,
        'char_entropy': char_entropy,
        'rep_rate': rep_rate,
        'punct_entropy': punct_entropy,
        'n_words': n_words,
        'n_sents': n_sents,
        'sent_lengths': sent_lengths,
        'text_lower': text_lower,
        'content_words': content_words,
    }


# ── Extended 20 features ──────────────────────────────────────────────────

def _extract_extended_features(text, words, sents, orig):
    n_words = len(words)
    n_sents = len(sents)
    words_per_1000 = n_words / 1000.0
    sent_lengths = orig['sent_lengths']
    text_lower = orig['text_lower']

    feats = {}

    # ── Readability (4 features) ──
    syllable_counts = [count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    if n_words > 0 and n_sents > 0:
        flesch_reading_ease = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (total_syllables / n_words)
        flesch_kincaid_grade = 0.39 * (n_words / n_sents) + 11.8 * (total_syllables / n_words) - 15.59
        gunning_fog = 0.4 * ((n_words / n_sents) + 100 * (complex_words / n_words))
    else:
        flesch_reading_ease = 0.0
        flesch_kincaid_grade = 0.0
        gunning_fog = 0.0

    if n_sents > 0:
        smog = 1.0430 * math.sqrt(complex_words * (30 / max(n_sents, 1))) + 3.1291
    else:
        smog = 0.0

    feats['flesch_reading_ease'] = flesch_reading_ease
    feats['flesch_kincaid_grade'] = flesch_kincaid_grade
    feats['gunning_fog'] = gunning_fog
    feats['smog_index'] = smog

    # ── Sentiment (5 features) ──
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos_density = pos_count / max(words_per_1000, 0.001)
    neg_density = neg_count / max(words_per_1000, 0.001)
    total_sent = pos_count + neg_count
    sentiment_polarity = (pos_count - neg_count) / max(total_sent, 1)

    excl_count = text.count('!')
    excl_density = excl_count / max(words_per_1000, 0.001)
    quest_count = text.count('?')
    quest_density = quest_count / max(words_per_1000, 0.001)

    feats['positive_density'] = pos_density
    feats['negative_density'] = neg_density
    feats['sentiment_polarity'] = sentiment_polarity
    feats['exclamation_density'] = excl_density
    feats['question_density'] = quest_density

    # ── Syntactic complexity (6 features) ──
    passive_matches = len(PASSIVE_INDICATORS.findall(text))
    passive_density = passive_matches / max(words_per_1000, 0.001)

    sub_count = sum(1 for w in words if w in SUBORDINATING_CONJUNCTIONS)
    sub_density = sub_count / max(words_per_1000, 0.001)

    prep_count = sum(1 for w in words if w in PREPOSITIONS)
    prep_density = prep_count / max(words_per_1000, 0.001)

    adj_count = sum(1 for w in words if any(w.endswith(suf) for suf in COMMON_ADJECTIVE_SUFFIXES) and len(w) > 4)
    adj_density = adj_count / max(words_per_1000, 0.001)

    adv_count = sum(1 for w in words if w.endswith('ly') and len(w) > 3)
    adv_density = adv_count / max(words_per_1000, 0.001)

    nom_count = sum(1 for w in words if any(w.endswith(suf) for suf in NOMINALIZATION_SUFFIXES) and len(w) > 5)
    nom_density = nom_count / max(words_per_1000, 0.001)

    feats['passive_density'] = passive_density
    feats['subordination_density'] = sub_density
    feats['preposition_density'] = prep_density
    feats['adjective_density'] = adj_density
    feats['adverb_density'] = adv_density
    feats['nominalization_density'] = nom_density

    # ── Named entities / surface features (5 features) ──
    cap_entities = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    cap_density = cap_entities / max(words_per_1000, 0.001)

    number_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
    number_density = number_count / max(words_per_1000, 0.001)

    acronyms = len(re.findall(r'\b[A-Z]{2,}(?:s)?\b', text))
    acronym_density = acronyms / max(words_per_1000, 0.001)

    url_email = len(re.findall(r'https?://\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b', text))
    url_email_density = url_email / max(words_per_1000, 0.001)

    quote_count = len(re.findall(r'"[^"]*"|\'[^\']*\'|"[^"]*"', text))
    quote_density = quote_count / max(words_per_1000, 0.001)

    feats['capitalized_entity_density'] = cap_density
    feats['number_density'] = number_density
    feats['acronym_density'] = acronym_density
    feats['url_email_density'] = url_email_density
    feats['quote_density'] = quote_density

    # ── Vocabulary richness (4 features) ──
    content_words = orig['content_words']
    if content_words:
        word_freq = Counter(content_words)
        n_words = len(content_words)
        n_types = len(word_freq)
        
        # Type-token ratio
        feats['type_token_ratio'] = n_types / max(n_words, 1)
        
        # Hapax legomena ratio (words appearing exactly once)
        hapax_count = sum(1 for c in word_freq.values() if c == 1)
        feats['hapax_legomena_ratio'] = hapax_count / max(n_words, 1)
        
        # Yule's K (lexical diversity)
        if n_words > 0:
            m1 = sum(c for c in word_freq.values())
            m2 = sum(c * c for c in word_freq.values())
            feats['yules_k'] = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0
        else:
            feats['yules_k'] = 0.0
        
        # Simpson's D (lexical diversity)
        if n_words > 1:
            feats['simpsons_d'] = 1 - sum((c / n_words) ** 2 for c in word_freq.values())
        else:
            feats['simpsons_d'] = 0.0
    else:
        feats['type_token_ratio'] = 0.0
        feats['hapax_legomena_ratio'] = 0.0
        feats['yules_k'] = 0.0
        feats['simpsons_d'] = 0.0

    return feats


# ── Public API ────────────────────────────────────────────────────────────

ORIGINAL_FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

EXTENDED_FEATURE_COLS = [
    'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'smog_index',
    'positive_density', 'negative_density', 'sentiment_polarity',
    'exclamation_density', 'question_density',
    'passive_density', 'subordination_density', 'preposition_density',
    'adjective_density', 'adverb_density', 'nominalization_density',
    'capitalized_entity_density', 'number_density', 'acronym_density',
    'url_email_density', 'quote_density',
    'type_token_ratio', 'hapax_legomena_ratio', 'yules_k', 'simpsons_d',
]

ALL_FEATURE_COLS = ORIGINAL_FEATURE_COLS + EXTENDED_FEATURE_COLS


def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)


def extract_features(text, extended=True, lang=None):
    """Extract stylometric features from a single text.

    Args:
        text: Input text string.
        extended: If True, extract all 31 features. If False, only original 11.
        lang: Language override (e.g. 'en', 'fr'). If None, dynamically detected.

    Returns:
        Dict of feature name -> float value, or None if text is too short.
    """
    if not text or not isinstance(text, str):
        return None

    text = normalize_unicode(text)
    words = tokenize_words(text)
    if len(words) < 5:
        return None

    if lang is None:
        from tool.multilingual import detect_language
        lang = detect_language(text)

    sents = tokenize_sentences(text)

    orig = _extract_original_features(text, words, sents, lang=lang)
    if orig is None:
        return None

    result = {k: v for k, v in orig.items() if k not in ('sent_lengths', 'text_lower', 'content_words')}

    if extended:
        ext = _extract_extended_features(text, words, sents, orig)
        result.update(ext)

    return result


def extract_feature_vector(text, feature_cols=None, extended=True, lang=None):
    """Extract features as an ordered list matching feature_cols.

    Args:
        text: Input text string.
        feature_cols: List of feature names to extract. If None, uses all.
        extended: Whether to compute extended features.
        lang: Language override. If None, dynamically detected.

    Returns:
        List of float values in the same order as feature_cols, or None.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS if extended else ORIGINAL_FEATURE_COLS

    feats = extract_features(text, extended=extended, lang=lang)
    if feats is None:
        return None
    return [feats.get(c, 0.0) for c in feature_cols]
