"""Unit tests for FallbackProvider routing decisions."""
from llamaherd.proxy import FallbackProvider, VALID_FALLBACK_PRIORITIES


def _provider(**overrides):
    cfg = {
        "provider": "nvidia-build",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-test",
        "default_model": "deepseek-ai/deepseek-v4-flash",
        "priority": "after",
        "model_map": {
            "glm-5.1": "z-ai/glm-5.1",
            "glm5": {"nvidia_model": "z-ai/glm5", "priority": "before"},
            "kimi-k2.6": {"nvidia_model": "moonshotai/kimi-k2.6"},
        },
    }
    cfg.update(overrides)
    return FallbackProvider(cfg)


def test_disabled_when_missing_credentials():
    fp = FallbackProvider({"provider": "nvidia-build"})
    assert fp.enabled is False


def test_resolve_model_string_shorthand():
    fp = _provider()
    assert fp.resolve_model("glm-5.1") == "z-ai/glm-5.1"


def test_resolve_model_dict_form():
    fp = _provider()
    assert fp.resolve_model("glm5") == "z-ai/glm5"
    assert fp.resolve_model("kimi-k2.6") == "moonshotai/kimi-k2.6"


def test_resolve_model_unmapped():
    fp = _provider()
    assert fp.resolve_model("totally-unknown-model") is None


def test_priority_for_global_default():
    fp = _provider()
    assert fp.priority_for("glm-5.1") == "after"
    assert fp.priority_for("kimi-k2.6") == "after"


def test_priority_for_per_model_override():
    fp = _provider()
    assert fp.priority_for("glm5") == "before"


def test_invalid_priority_falls_back_to_after():
    fp = _provider(priority="weird")
    assert fp.priority == "after"


def test_set_priority_updates_global():
    fp = _provider()
    assert fp.set_priority("only") == "only"
    assert fp.priority == "only"
    fp.set_priority("invalid")
    assert fp.priority == "only"  # unchanged


def test_should_try_only_priority():
    fp = _provider()
    assert fp.should_try("only", model_available_on_ollama=True) is True
    assert fp.should_try("only", model_available_on_ollama=False) is True


def test_should_try_before_priority():
    fp = _provider()
    assert fp.should_try("before", model_available_on_ollama=True) is True


def test_should_try_after_priority():
    fp = _provider()
    assert fp.should_try("after", model_available_on_ollama=True) is False
    assert fp.should_try("after", model_available_on_ollama=False) is True


def test_disabled_provider_never_serves():
    fp = FallbackProvider({})
    assert fp.should_try("before", model_available_on_ollama=False) is False
    assert fp.should_try("only", model_available_on_ollama=False) is False


def test_model_aliases_includes_priority():
    fp = _provider()
    aliases = {a["id"]: a for a in fp.model_aliases()}
    assert aliases["glm-5.1"]["nvidia_model"] == "z-ai/glm-5.1"
    assert aliases["glm-5.1"]["priority"] == "after"  # global default
    assert aliases["glm5"]["priority"] == "before"  # per-model override


def test_valid_priorities_const():
    assert set(VALID_FALLBACK_PRIORITIES) == {"after", "before", "only"}
