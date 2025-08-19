"""
SPEECH_KEY=<secret> uv run python test/azure_test_s2tt.py
"""

import os

import azure.cognitiveservices.speech as speechsdk


def recognize_from_microphone() -> None:
    """
    This example requires environment variables named "SPEECH_KEY" and "ENDPOINT"

    Replace with your own subscription key and endpoint, the endpoint is like:
    "https://YourServiceRegion.api.cognitive.microsoft.com"
    """
    endpoint = "https://northeurope.api.cognitive.microsoft.com/"
    speech_translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=os.environ.get("SPEECH_KEY"), endpoint=endpoint
    )
    speech_translation_config.speech_recognition_language = "da-DK"

    to_language = "en"
    speech_translation_config.add_target_language(to_language)

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    translation_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=speech_translation_config, audio_config=audio_config
    )

    print("Speak into your microphone.")
    translation_recognition_result = translation_recognizer.recognize_once_async().get()
    assert translation_recognition_result

    if translation_recognition_result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(f"Recognized: {translation_recognition_result.text}")
        print(f"""Translated into '{to_language}': {translation_recognition_result.translations[to_language]}""")
    elif translation_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print(f"No speech could be recognized: {translation_recognition_result.no_match_details}")
    elif translation_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = translation_recognition_result.cancellation_details
        print(f"Speech Recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
            print("Did you set the speech resource key and endpoint values?")


recognize_from_microphone()
