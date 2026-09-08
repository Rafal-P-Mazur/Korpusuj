from korpusuj.corpus.creator_orchestration import (
    _NLP_UNSAFE_FORMAT_TRANSLATION,
    _UNICODE_NORMALIZATION_STATS,
    _normalize_creator_text_for_nlp,
    _reset_unicode_normalization_stats,
)


def test_explicit_unicode_normalization_184d():
    source = "a\u00adb c\u200bd e\u200cf g\u200dh i\u200ej k\u2060l m\u2063n o\u2066p"
    assert _normalize_creator_text_for_nlp(source) == "ab cd ef gh ij kl mn op"
    assert len(_NLP_UNSAFE_FORMAT_TRANSLATION) == 8


def test_structure_and_polish_text_are_preserved_184d():
    source = "Zażółć\tgęślą.\n\nukraiń\u200bskiego — «tekst»  z dwiema spacjami"
    expected = "Zażółć\tgęślą.\n\nukraińskiego — «tekst»  z dwiema spacjami"
    assert _normalize_creator_text_for_nlp(source) == expected


def test_double_zero_width_regression_and_aggregate_184d():
    _reset_unicode_normalization_stats()
    assert _normalize_creator_text_for_nlp("fragment\u200b\u200bkońcowy") == "fragmentkońcowy"
    assert _UNICODE_NORMALIZATION_STATS == {"documents": 1, "characters": 2}
