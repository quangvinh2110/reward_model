#!/usr/bin/env python3
"""
Test script for the perplexity computation module.

This script demonstrates how to use the perplexity calculator with both
direct Hugging Face models and vLLM API.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from scoring.perplexify import (
    PerplexityCalculator,
    PerplexityConfig,
    compute_perplexity_batch,
    create_quantization_config,
)


def test_direct_hf_model():
    """Test perplexity computation with direct Hugging Face model."""
    print("=== Testing Direct Hugging Face Model ===")

    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather is beautiful today.",
        "Mathematics is the language of the universe.",
    ]

    try:
        # Create configuration for direct model loading
        config = PerplexityConfig(
            model_name="gpt2",  # Use a smaller model for testing
            max_length=1024,
            stride=256,
            device="auto",
            use_vllm=False,
            quantization_config=None,  # Disable quantization for testing
        )

        # Create calculator
        calculator = PerplexityCalculator(config)

        # Compute perplexity
        ppl_scores = calculator.compute_perplexity(texts)

        print("Perplexity scores:")
        for i, (text, ppl) in enumerate(zip(texts, ppl_scores)):
            print(f"  {i+1}. Text: {text[:50]}...")
            print(f"     Perplexity: {ppl:.4f}")

        return True

    except Exception as e:
        print(f"Error testing direct HF model: {e}")
        return False


def test_vllm_api():
    """Test perplexity computation with vLLM API."""
    print("\n=== Testing vLLM API ===")

    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    try:
        # Create configuration for vLLM API
        config = PerplexityConfig(
            model_name="gpt2",
            use_vllm=True,
            vllm_api_url="http://localhost:8000",  # Adjust this URL as needed
        )

        # Create calculator
        calculator = PerplexityCalculator(config)

        # Compute perplexity
        ppl_scores = calculator.compute_perplexity(texts)

        print("vLLM Perplexity scores:")
        for i, (text, ppl) in enumerate(zip(texts, ppl_scores)):
            print(f"  {i+1}. Text: {text[:50]}...")
            print(f"     Perplexity: {ppl:.4f}")

        return True

    except Exception as e:
        print(f"Error testing vLLM API: {e}")
        print("Note: Make sure vLLM server is running on localhost:8000")
        return False


def test_convenience_function():
    """Test the convenience function for batch processing."""
    print("\n=== Testing Convenience Function ===")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    try:
        # Use convenience function
        ppl_scores = compute_perplexity_batch(
            texts=texts,
            model_name="gpt2",
            use_vllm=False,
            load_in_4bit=False,  # Disable quantization for testing
            max_length=1024,
            stride=256,
        )

        print("Convenience function results:")
        for i, (text, ppl) in enumerate(zip(texts, ppl_scores)):
            print(f"  {i+1}. Text: {text[:50]}...")
            print(f"     Perplexity: {ppl:.4f}")

        return True

    except Exception as e:
        print(f"Error testing convenience function: {e}")
        return False


def test_quantization():
    """Test model loading with quantization."""
    print("\n=== Testing Quantization ===")

    try:
        # Create quantization config
        quantization_config = create_quantization_config(load_in_4bit=True)

        if quantization_config:
            print("Quantization config created successfully")
            print(f"  load_in_4bit: {quantization_config.load_in_4bit}")
            print(f"  load_in_8bit: {quantization_config.load_in_8bit}")
        else:
            print("No quantization config created")

        return True

    except Exception as e:
        print(f"Error testing quantization: {e}")
        return False


def main():
    """Main test function."""
    print("Perplexity Computation Module Test")
    print("=" * 50)

    # Test direct Hugging Face model
    test_direct_hf_model()

    # Test vLLM API (will fail if server not running)
    test_vllm_api()

    # Test convenience function
    test_convenience_function()

    # Test quantization config
    test_quantization()

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()
