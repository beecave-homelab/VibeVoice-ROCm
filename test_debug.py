#!/usr/bin/env python3
"""
Test script to demonstrate the improved AI script generation with debug output.
This shows what prompts are being sent to OpenAI and how context is processed.
"""

import os
import sys
import random

# Add the demo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'demo'))

# Mock OpenAI for testing
class MockOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key

    class MockCompletions:
        def create(self, **kwargs):
            # Mock response
            class MockChoice:
                class MockMessage:
                    content = """Alice: That's a fascinating scenario about time travel!
Carter: I agree, the implications for causality are mind-bending.
Alice: Exactly! What if we could witness historical events firsthand?
Carter: The technology would revolutionize education and research."""

                message = MockMessage()

            class MockUsage:
                total_tokens = 150

            class MockResponse:
                choices = [MockChoice()]
                usage = MockUsage()

            return MockResponse()

    chat = MockCompletions()

# Test the improved script generation logic
def test_script_generation():
    print("🧪 Testing Improved AI Script Generation\n")

    # Mock input scenario
    test_scenario = """Alex: I've been thinking about the ethics of artificial intelligence.
Jordan: That's a complex topic. What aspects concern you most?
Alex: The potential for bias in decision-making systems."""

    print("📝 Test Scenario:")
    print(test_scenario)
    print("\n" + "="*50 + "\n")

    # Simulate the improved topic extraction
    lines = test_scenario.strip().split('\n')
    meaningful_lines = []

    for line in lines:
        line = line.strip()
        if ':' in line and len(line.split(':', 1)[1].strip()) > 3:
            meaningful_lines.append(line)

    context_description = '\n'.join(meaningful_lines)
    all_content = ' '.join([line.split(':', 1)[1].strip() for line in meaningful_lines])
    words = all_content.split()

    # Improved topic extraction with stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    meaningful_words = []
    for word in words[:15]:
        if len(word) > 2 and word.lower() not in stop_words:
            meaningful_words.append(word.lower())
    topic = ' '.join(meaningful_words[:6])

    print("🔍 DEBUG: Script analysis complete")
    print(f"🔍 DEBUG: Extracted topic: '{topic}'")
    print(f"🔍 DEBUG: Full context: {context_description}")

    # Simulate speaker name extraction
    speaker_names = ['Alex', 'Jordan']
    num_speakers = 2
    conversation_style = "casual, friendly conversation"

    print(f"🔍 DEBUG: Speaker names: {speaker_names}")
    print(f"🔍 DEBUG: Number of speakers: {num_speakers}")
    print(f"🔍 DEBUG: Style: {conversation_style}")

    # Build the improved prompt
    prompt_parts = []

    if speaker_names:
        speaker_info = ", ".join(speaker_names[:num_speakers])
        prompt_parts.append(f"Generate a conversation between: {speaker_info}")
    else:
        prompt_parts.append(f"Generate a {num_speakers}-person conversation")

    prompt_parts.append(f"Style: {conversation_style}")

    if context_description.strip():
        prompt_parts.append(f"IMPORTANT CONTEXT: Use the following as the basis for the conversation:")
        prompt_parts.append(f"{context_description}")
        if topic:
            prompt_parts.append(f"Key topic elements: {topic}")
        prompt_parts.append("Continue this conversation naturally based on the context provided above.")
    else:
        prompt_parts.append(f"Topic: {topic}")

    if speaker_names and len(speaker_names) >= num_speakers:
        speaker_format = " or ".join([f"'{name}:'" for name in speaker_names[:num_speakers]])
        prompt_parts.append(f"Format: Each line should start with {speaker_format}")
    else:
        speaker_list = [f"Speaker {i}:" for i in range(num_speakers)]
        speaker_format = " or ".join([f"'{s}'" for s in speaker_list])
        prompt_parts.append(f"Format: Each line should start with {speaker_format}")

    prompt_parts.append("Keep it conversational, engaging, and around 4-8 exchanges total.")
    prompt_parts.append("Make it sound natural and authentic.")
    prompt_parts.append("End with a satisfying conclusion.")

    prompt = "\n".join(prompt_parts)

    print("\n🔍 DEBUG: Generated prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    # Simulate API call
    print("🔍 DEBUG: Sending request to OpenAI API...")
    print("🔍 DEBUG: Model: gpt-4o-mini")
    print("🔍 DEBUG: Max tokens: 600")
    print("🔍 DEBUG: Temperature: 0.8")

    # Mock response
    mock_response = """Alex: That's a fascinating scenario about time travel!
Carter: I agree, the implications for causality are mind-bending.
Alex: Exactly! What if we could witness historical events firsthand?
Carter: The technology would revolutionize education and research."""

    print("🔍 DEBUG: Received response from OpenAI API")
    print("🔍 DEBUG: Response tokens used: 150")
    print(f"🔍 DEBUG: Raw response content: {mock_response[:200]}...")

    print("\n✨ Final Result:")
    print("-" * 30)
    print(mock_response)
    print("-" * 30)

if __name__ == "__main__":
    test_script_generation()
