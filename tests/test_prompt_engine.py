
import pytest

from ravi_varma.generation.prompt_engine import (
    DEFAULT_NEGATIVE_PROMPT,
    LocalPromptEngine,
    PromptExpansionEngine,
)

SCENE = "Yudhishthira weeping over the fallen body of Karna on the battlefield of Kurukshetra."


class TestTriggerTokenInsertion:
    def test_trigger_token_present_when_style_enabled(self):
        engine = PromptExpansionEngine(trigger_token="<rvvarma>")
        result = engine.expand(SCENE, enable_style=True)
        assert result.final_prompt().startswith("<rvvarma>,")

    def test_trigger_token_absent_when_style_disabled(self):
        engine = PromptExpansionEngine(trigger_token="<rvvarma>")
        result = engine.expand(SCENE, enable_style=False)
        assert "<rvvarma>" not in result.final_prompt()


class TestUserMeaningPreservation:
    def test_subject_preserves_full_scene_text(self):
        result = LocalPromptEngine().expand(SCENE)
        assert result.subject == SCENE

    def test_final_prompt_contains_scene_verbatim(self):
        result = LocalPromptEngine().expand(SCENE, trigger_token="<rvvarma>")
        assert SCENE in result.final_prompt()

    def test_empty_scene_raises(self):
        with pytest.raises(ValueError):
            LocalPromptEngine().expand("   ")

    def test_does_not_invent_environment_when_no_keyword_present(self):
        result = LocalPromptEngine().expand("A quiet conversation between two sisters.")
        assert "unspecified" in result.environment

    def test_extracts_known_environment_keyword(self):
        result = LocalPromptEngine().expand(SCENE)  # contains "battlefield"
        assert "battlefield" in result.environment


class TestNegativePromptCreation:
    def test_default_negative_prompt_used_when_not_overridden(self):
        result = LocalPromptEngine().expand(SCENE)
        assert result.negative_prompt == DEFAULT_NEGATIVE_PROMPT

    def test_custom_negative_prompt_overrides_default(self):
        custom = "low quality, watermark"
        result = LocalPromptEngine().expand(SCENE, negative_prompt=custom)
        assert result.negative_prompt == custom

    def test_negative_prompt_covers_common_anatomy_issues(self):
        assert "extra fingers" in DEFAULT_NEGATIVE_PROMPT
        assert "deformed hands" in DEFAULT_NEGATIVE_PROMPT


class TestLocalFallback:
    def test_falls_back_to_local_when_no_provider_configured(self, monkeypatch):
        monkeypatch.delenv("PROMPT_EXPANSION_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        engine = PromptExpansionEngine(trigger_token="<rvvarma>")
        result = engine.expand(SCENE)
        assert result.mode_used == "local"

    def test_falls_back_to_local_when_llm_call_raises(self, monkeypatch):
        monkeypatch.setenv("PROMPT_EXPANSION_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-real-key")
        engine = PromptExpansionEngine(trigger_token="<rvvarma>")
        # No real network access / invalid key -> the OpenAI call will fail
        # (ImportError if openai isn't installed, or an API error otherwise)
        # and PromptExpansionEngine must transparently fall back to local.
        result = engine.expand(SCENE)
        assert result.mode_used == "local"
        assert SCENE in result.final_prompt()

    def test_force_local_skips_llm_even_if_configured(self, monkeypatch):
        monkeypatch.setenv("PROMPT_EXPANSION_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        engine = PromptExpansionEngine(trigger_token="<rvvarma>")
        result = engine.expand(SCENE, force_local=True)
        assert result.mode_used == "local"

    def test_llm_output_is_grounded_in_lexicon(self, monkeypatch):
        from ravi_varma.generation.prompt_engine import LLMPromptEngine

        monkeypatch.setenv("PROMPT_EXPANSION_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        sent = []

        def fake_call(self, message):
            sent.append(message)
            return {"subject": f"{SCENE} rewritten at length", "environment": ["dusty plain", "banners"], "negative_prompt": "llm neg"}

        monkeypatch.setattr(LLMPromptEngine, "_call_openai", fake_call)
        result = PromptExpansionEngine(trigger_token="<rvvarma>", negative_prompt="cfg neg").expand(SCENE)
        assert result.mode_used == "llm"
        assert "golden armour" in sent[0]  # Karna's lexicon look is given to the LLM as context
        assert result.subject == SCENE  # scene kept verbatim, LLM rewording dropped
        assert any("Karna" in d for d in result.details)
        assert result.environment == "dusty plain, banners"
        assert result.negative_prompt == "cfg neg"


class TestSemanticExpansion:
    """Milestone 3: lexicon-grounded detail within CLIP's token window."""

    @staticmethod
    def words(text):  # deterministic stand-in for the CLIP tokenizer
        return len(text.replace(",", " ").split())

    def test_character_iconography_added_in_scene_order(self):
        result = LocalPromptEngine().expand("Krishna guiding Arjuna in his chariot", trigger_token="<rvvarma>")
        assert result.characters == "Krishna, Arjuna"
        assert "peacock-feather" in result.details[0] and "bow" in result.details[1]

    def test_alias_and_specific_place_win(self):
        result = LocalPromptEngine().expand("Droupathi pleading in the court of Hastinapura")
        assert result.characters == "Draupadi"
        assert "Hastinapura" in result.environment  # specific place beats generic "court"

    def test_prompt_trimmed_to_budget_but_scene_kept(self):
        result = LocalPromptEngine().expand(SCENE, trigger_token="<rvvarma>")
        prompt = result.final_prompt(token_counter=self.words, max_tokens=30)
        assert self.words(prompt) <= 30
        assert prompt.startswith("<rvvarma>, " + SCENE)

    def test_unknown_names_add_no_invented_detail(self):
        result = LocalPromptEngine().expand("A quiet conversation between two sisters")
        assert result.characters == "" and result.details == []
