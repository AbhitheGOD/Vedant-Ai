import re
import asyncio
from typing import Tuple
from langdetect import detect, LangDetectException

try:
    from googletrans import Translator as GTranslator
    _gtrans = GTranslator()
except Exception:
    _gtrans = None


# Mapping of ISO 639-1 codes to human-readable names and native names
LANGUAGE_INFO: dict = {
    "en": ("English", "English"),
    "hi": ("Hindi", "हिन्दी"),
    "sa": ("Sanskrit", "संस्कृतम्"),
    "ta": ("Tamil", "தமிழ்"),
    "te": ("Telugu", "తెలుగు"),
    "kn": ("Kannada", "ಕನ್ನಡ"),
    "ml": ("Malayalam", "മലയാളം"),
    "mr": ("Marathi", "मराठी"),
    "bn": ("Bengali", "বাংলা"),
    "gu": ("Gujarati", "ગુજરાતી"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ"),
    "or": ("Odia", "ଓଡ଼ିଆ"),
    "ur": ("Urdu", "اردو"),
}

# Devanagari Unicode range: U+0900–U+097F
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def _is_devanagari(text: str) -> bool:
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(text))
    return devanagari_chars / max(len(text), 1) > 0.3


def _has_iast_markers(text: str) -> bool:
    """Heuristic: IAST transliteration markers common in Sanskrit."""
    iast_chars = re.findall(r"[āīūṛṝḷṃḥśṣṭḍṇñṅ]", text, re.IGNORECASE)
    return len(iast_chars) >= 2


def detect_language(text: str) -> str:
    """
    Detect language and return ISO 639-1 code.
    Adds heuristics for Sanskrit (langdetect is poor at it).
    """
    if not text or not text.strip():
        return "en"

    # Heuristic: if heavy Devanagari, try to distinguish Hindi vs Sanskrit
    if _is_devanagari(text):
        # Sanskrit-specific patterns: vibhakti endings, sandhi markers
        sanskrit_markers = re.compile(
            r"\b(अस्ति|भवति|कः|किम्|यत्र|तत्र|सर्वम्|धर्मः|कर्म|योगः|ज्ञानम्|आत्मा|ब्रह्म)\b"
        )
        if sanskrit_markers.search(text):
            return "sa"
        try:
            lang = detect(text)
            return lang if lang in LANGUAGE_INFO else "hi"
        except LangDetectException:
            return "hi"

    if _has_iast_markers(text):
        return "sa"

    try:
        lang = detect(text)
        # Map some edge cases
        if lang == "zh-cn" or lang == "zh-tw":
            return "zh"
        return lang
    except LangDetectException:
        return "en"


async def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English using Google Translate. Returns original if already English."""
    if source_lang == "en":
        return text

    if _gtrans is not None:
        try:
            result = await asyncio.to_thread(_gtrans.translate, text, dest="en", src=source_lang)
            return result.text.strip()
        except Exception:
            pass

    # Fallback: use LLM for translation
    from app.services.llm_service import get_llm_service
    lang_name = LANGUAGE_INFO.get(source_lang, (source_lang, source_lang))[0]
    llm = get_llm_service()
    try:
        translated = await llm.generate(
            prompt=f"Translate this {lang_name} text to English:\n\n{text}",
            system_prompt="You are a precise translator. Output only the translation.",
        )
        return translated.strip()
    except Exception:
        return text


def get_response_language_instruction(lang_code: str) -> str:
    """Returns the instruction string to embed in the system prompt."""
    if lang_code == "en":
        return "Respond in English."

    info = LANGUAGE_INFO.get(lang_code)
    if info:
        lang_name, native_name = info
        return f"Respond in {lang_name} ({native_name}). Use the {native_name} script where appropriate."

    return f"Respond in the same language as the user's query (language code: {lang_code})."


def get_language_display_name(lang_code: str) -> str:
    info = LANGUAGE_INFO.get(lang_code)
    return info[0] if info else lang_code
