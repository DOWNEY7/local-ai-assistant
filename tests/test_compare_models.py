import pytest
from unittest.mock import patch, MagicMock
from src.compare_models import benchmark_single

def mock_ollama_stream_success(*args, **kwargs):
    """Mocks a successful streaming response from Ollama."""
    chunks = [
        {"message": {"content": "Hello"}},
        {"message": {"content": " World"}},
        {
            "message": {"content": "!"},
            "eval_count": 10,
            "eval_duration": 1000000000  # 1 second in nanoseconds
        }
    ]
    for chunk in chunks:
        yield chunk

@patch("src.compare_models.ollama.chat")
def test_benchmark_single_success(mock_chat):
    # Setup mock to return our generator
    mock_chat.return_value = mock_ollama_stream_success()
    
    # Run the function
    result = benchmark_single("test-model", "test prompt")
    
    # Assertions
    assert result["tokens"] == 10
    assert result["tps"] == 10.0  # 10 tokens / 1 second
    assert result["ttft"] is not None
    assert isinstance(result["ttft"], float)
    
@patch("src.compare_models.ollama.chat")
def test_benchmark_single_no_metadata(mock_chat):
    """Test when Ollama doesn't return evaluation metadata (e.g., error or early abort)."""
    def mock_stream_no_meta():
        yield {"message": {"content": "Just text"}}
        
    mock_chat.return_value = mock_stream_no_meta()
    
    result = benchmark_single("test-model", "test prompt")
    
    assert result["tokens"] is None
    assert result["tps"] is None
    assert result["ttft"] is not None
