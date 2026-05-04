from llamaherd.proxy import fmt_param_count


def test_billion_clean():
    assert fmt_param_count(8_000_000_000) == "8B"
    assert fmt_param_count(70_000_000_000) == "70B"


def test_billion_decimal():
    assert fmt_param_count(8_300_000_000) == "8.3B"


def test_trillion():
    assert fmt_param_count(1_700_000_000_000) == "1.7T"
    assert fmt_param_count(1_000_000_000_000) == "1T"


def test_edge_cases():
    assert fmt_param_count(None) == ""
    assert fmt_param_count(0) == ""
    assert fmt_param_count("not-a-number") == ""
