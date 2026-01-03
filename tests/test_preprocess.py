from scam_detection.data.preprocessing import clean_text, load_and_preprocess_data


def test_clean_text():
    text = "Hello, World! 123"
    cleaned = clean_text(text)
    assert cleaned == "hello world 123"


def test_load_and_preprocess_data():
    df = load_and_preprocess_data("tests/fixtures/tiny_dataset.csv")
    assert len(df) > 0
    assert "text" in df.columns
    assert "label" in df.columns
    assert df["label"].isin([0, 1]).all()
