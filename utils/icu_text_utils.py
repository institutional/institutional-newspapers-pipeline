def icu_word_tokenize(text: str, language_code: str) -> list[str]:
    """Language-aware word tokenization using ICU's BreakIterator."""
    import icu

    locale = icu.Locale(language_code)
    breaker = icu.BreakIterator.createWordInstance(locale)
    breaker.setText(text)

    words: list[str] = []
    start = 0
    for end in breaker:
        # Skip non-word segments (whitespace, punctuation)
        if breaker.getRuleStatus() > 0:
            word = text[start:end]
            if word.strip():
                words.append(word)
        start = end

    return words


def icu_sentence_tokenize(text: str, language_code: str) -> list[str]:
    """Language-aware sentence tokenization using ICU's BreakIterator."""
    import icu

    locale = icu.Locale(language_code)
    breaker = icu.BreakIterator.createSentenceInstance(locale)
    breaker.setText(text)

    sentences: list[str] = []
    start = 0
    for end in breaker:
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end

    return sentences


def iso639_3_to_1(code_3: str | None) -> str:
    """Converts ISO 639-3 to ISO 639-1 for ICU locale. Defaults to 'en'."""
    import pycountry

    if not code_3:
        return "en"
    try:
        lang = pycountry.languages.get(alpha_3=code_3)
        return lang.alpha_2 if lang and hasattr(lang, "alpha_2") else "en"
    except Exception:
        return "en"
