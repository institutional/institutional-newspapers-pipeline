import functools

from const import LANGUAGE_MIN_CONFIDENCE_SCORE, LANGUAGE_MIN_VLM_WORD_COUNT


@functools.cache
def _detectable_languages() -> frozenset[str]:
    """ISO 639-3 codes the detector can actually emit, in Lingua's uppercase form."""
    from lingua import Language

    return frozenset(language.iso_code_639_3.name for language in Language.all())


def postprocess_language_detection(
    language_code: str,
    confidence_score: float,
    vlm_word_count: int,
    issue_language: str | None,
) -> tuple[str, bool]:
    """Overrides detected language with issue-level language when confidence or word count is low."""
    if not issue_language:
        return (language_code, False)

    issue_language = issue_language.upper()  # Match Lingua format

    if issue_language not in _detectable_languages():
        return (issue_language, True)

    if confidence_score < LANGUAGE_MIN_CONFIDENCE_SCORE:
        return (issue_language, True)

    if vlm_word_count < LANGUAGE_MIN_VLM_WORD_COUNT:
        return (issue_language, True)

    return (language_code, False)
