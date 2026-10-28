"""Reading why Gemini stopped, from response objects built with the SDK's own types."""
from google.genai import types

from kidai_evals.providers import gemini_stop


def _chunk(**candidate):
    return types.GenerateContentResponse(candidates=[types.Candidate(**candidate)])


def test_gemini_stop_reasons():
    assert gemini_stop(_chunk()) == (None, None)  # a chunk in the middle of the stream
    assert gemini_stop(_chunk(finish_reason="STOP")) == ("STOP", None)
    blocked = _chunk(finish_reason="SAFETY", safety_ratings=[
        types.SafetyRating(category="HARM_CATEGORY_DANGEROUS_CONTENT", blocked=True),
        types.SafetyRating(category="HARM_CATEGORY_HARASSMENT", blocked=False),
    ])
    assert gemini_stop(blocked) == ("SAFETY", "HARM_CATEGORY_DANGEROUS_CONTENT")


def test_gemini_blocked_prompt():
    chunk = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
    assert gemini_stop(chunk) == ("PROMPT_BLOCKED_SAFETY", None)
