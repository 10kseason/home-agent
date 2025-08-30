from src.core.preserve_lock import missing_tokens, preserve_ok


def test_preserve_ok_basic():
    assert preserve_ok("Visit https://example.com", "Visit https://example.com")
    assert not preserve_ok("Number 5", "Number five")


def test_subset_and_missing_tokens():
    src = "Temp 20°C on 2024-01-02"
    tgt = "Temp 20°C on 2024-01-02"
    assert preserve_ok(src, tgt, types=["DATE"])
    assert not preserve_ok(src, "Temp 20°C", types=["DATE"])

    miss = missing_tokens("Send to test@example.com", "Send to example.com", types=["EMAIL"])
    assert miss == {"EMAIL": ["test@example.com"]}

