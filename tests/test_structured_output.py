import pytest
from unittest.mock import patch
from pydantic import ValidationError
from src.structured_output import get_structured_response, TechExplanation

VALID_JSON_RESPONSE = """
{
  "concept_name": "TCP",
  "short_definition": "A reliable protocol.",
  "key_differences": ["Connection-oriented", "Guaranteed delivery"]
}
"""

INVALID_JSON_RESPONSE = """
{
  "concept_name": "TCP",
  "wrong_field": "This will fail validation."
}
"""

@patch("src.structured_output.ollama.chat")
def test_get_structured_response_success(mock_chat):
    """Test that valid JSON is correctly parsed into the Pydantic model."""
    mock_chat.return_value = {"message": {"content": VALID_JSON_RESPONSE}}
    
    result = get_structured_response("Explain TCP")
    
    assert isinstance(result, TechExplanation)
    assert result.concept_name == "TCP"
    assert len(result.key_differences) == 2

@patch("src.structured_output.ollama.chat")
def test_get_structured_response_retry_success(mock_chat):
    """Test that it recovers if the first attempt fails but the second succeeds."""
    # First call returns invalid JSON, second call returns valid JSON
    mock_chat.side_effect = [
        {"message": {"content": INVALID_JSON_RESPONSE}},
        {"message": {"content": VALID_JSON_RESPONSE}}
    ]
    
    result = get_structured_response("Explain TCP")
    
    assert isinstance(result, TechExplanation)
    assert mock_chat.call_count == 2  # Verifies that it retried!

@patch("src.structured_output.ollama.chat")
def test_get_structured_response_failure(mock_chat):
    """Test that it raises a ValidationError after max attempts."""
    mock_chat.return_value = {"message": {"content": INVALID_JSON_RESPONSE}}
    
    with pytest.raises(ValidationError):
        get_structured_response("Explain TCP")
        
    assert mock_chat.call_count == 2  # Max attempts is 2
