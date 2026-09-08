from pathlib import Path


def _source():
    root = Path(__file__).resolve().parents[2]
    return (root / "korpusuj/corpus/creator_nlp.py").read_text(encoding="utf-8")


def test_herference_uses_mps_only_on_macos_when_available():
    source = _source()
    assert 'sys.platform == "darwin"' in source
    assert 'mps_backend.is_available()' in source
    assert 'herference_config = {"device": "mps"}' in source
    assert 'add_pipe("herference", config=herference_config)' in source


def test_non_mps_path_preserves_existing_default_behavior():
    source = _source()
    assert 'herference_config = {}' in source
    assert 'state.nlp_spacy.add_pipe("herference")' in source
    assert '[APP ml.device] component=herference selected=%s' in source


def test_stanza_and_easyocr_policy_is_not_changed_by_this_patch():
    source = _source()
    assert 'use_gpu=bool(torch.cuda.is_available())' in source
