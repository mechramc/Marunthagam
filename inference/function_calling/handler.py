"""
Parse and validate LLM function calling output for Marunthagam triage_classify().

Converts raw llama.cpp text output to validated Pydantic objects.
The disclaimer is always enforced at the schema level — never trust the model's disclaimer.
"""
import json
import re

from schemas import TriageClassifyInput, TriageClassifyOutput, DISCLAIMER_TEXT


# Gemma 4 / common tool call tag pattern
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def extract_function_call(raw_output: str) -> dict | None:
    """
    Extract function call JSON from raw model output.

    Tries <tool_call>...</tool_call> tags first (Gemma 4 format),
    then falls back to parsing the entire string as JSON.
    Returns None if no valid JSON is found.
    """
    match = _TOOL_CALL_PATTERN.search(raw_output)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing entire output as JSON
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None


def parse_triage_input(raw_args: dict) -> TriageClassifyInput:
    """
    Parse and validate triage_classify() input arguments.
    Raises pydantic.ValidationError on invalid data.
    """
    return TriageClassifyInput.model_validate(raw_args)


def parse_triage_output(raw_result: dict) -> TriageClassifyOutput:
    """
    Parse and validate triage_classify() return value.
    Always overwrites disclaimer with the Tamil string regardless of model output.
    """
    raw_result = dict(raw_result)  # copy to avoid mutating caller's dict
    raw_result["disclaimer"] = DISCLAIMER_TEXT
    return TriageClassifyOutput.model_validate(raw_result)
