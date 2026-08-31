from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pitch_artifacts_exist_and_use_r1_evidence():
    for name in ["PITCH_SCRIPT.md", "JUDGE_QA.md", "PRESENTATION_METRICS.md", "ARCHITECTURE_PITCH.md"]:
        text = (ROOT / name).read_text()
        assert "model_f_r1" in text or "Model F-R1" in text
        assert "82e77daac0762a04" not in text or "Do not" in text
        assert "live production" not in text.lower() or "not" in text.lower()


def test_judge_qa_has_broad_coverage():
    text = (ROOT / "JUDGE_QA.md").read_text()
    assert text.count("## ") >= 20
    for phrase in ["Why graph", "false positives", "production ready", "Razorpay", "R1"]:
        assert phrase.lower() in text.lower()


def test_presentation_metrics_has_provenance_and_status():
    text = (ROOT / "PRESENTATION_METRICS.md").read_text()
    assert "held-out" in text
    assert "synthetic" in text.lower()
    assert "3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff" in text
    assert "Do not use" in text
