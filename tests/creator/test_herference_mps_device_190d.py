from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CREATOR_NLP = ROOT / "korpusuj" / "corpus" / "creator_nlp.py"
CREATOR = ROOT / "korpusuj" / "corpus" / "creator.py"
CREATOR_ORCHESTRATION = ROOT / "korpusuj" / "corpus" / "creator_orchestration.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_herference_uses_mps_only_on_macos_when_available():
    source = _source(CREATOR_NLP)
    assert 'sys.platform == "darwin"' in source
    assert 'mps_backend.is_available()' in source
    assert 'herference_config = {"device": "mps"}' in source
    assert 'add_pipe("herference", config=herference_config)' in source


def test_non_mps_path_preserves_existing_default_behavior():
    source = _source(CREATOR_NLP)
    assert 'herference_config = {}' in source
    assert 'state.nlp_spacy.add_pipe("herference")' in source
    assert '[APP ml.device] component=herference selected=%s' in source


def test_stanza_keeps_cuda_availability_policy():
    source = _source(CREATOR_NLP)
    assert '"use_gpu": bool(torch.cuda.is_available())' in source


def test_easyocr_keeps_cuda_availability_policy():
    creator_source = _source(CREATOR)
    orchestration_source = _source(CREATOR_ORCHESTRATION)
    expected = 'gpu=bool(torch.cuda.is_available())'
    assert expected in creator_source
    assert expected in orchestration_source
