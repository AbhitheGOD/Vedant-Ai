from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="VedantAI", version="1.0.0")


@dataclass
class KnowledgeEntry:
    concept: str
    keywords: List[str]
    summary: str
    sanskrit_quote: str
    source: str
    recommendation: str


KNOWLEDGE_BASE: List[KnowledgeEntry] = [
    KnowledgeEntry(
        concept="karma",
        keywords=["karma", "action", "deed", "consequence", "karm"],
        summary=(
            "Karma in Vedic thought means intentional action and its ethical result. "
            "Actions done with attachment bind the mind, while selfless duty (nishkama karma) "
            "supports inner freedom."
        ),
        sanskrit_quote="कर्मण्येवाधिकारस्ते मा फलेषु कदाचन (Bhagavad Gita 2.47)",
        source="Bhagavad Gita, Chapter 2",
        recommendation="You may also ask about dharma and moksha for deeper context.",
    ),
    KnowledgeEntry(
        concept="dharma",
        keywords=["dharma", "duty", "righteous", "ethics", "order"],
        summary=(
            "Dharma is the principle of right living, responsibility, and cosmic order. "
            "It can vary by context but always points toward truth, harmony, and welfare."
        ),
        sanskrit_quote="धर्मो रक्षति रक्षितः",
        source="Manusmriti (traditional maxim)",
        recommendation="Try asking how dharma relates to karma in daily life.",
    ),
    KnowledgeEntry(
        concept="atman",
        keywords=["atman", "self", "soul", "brahman", "consciousness"],
        summary=(
            "Atman is the innermost self, beyond body and mind. Upanishadic teaching "
            "describes realization of Atman as the key to liberation."
        ),
        sanskrit_quote="अहं ब्रह्मास्मि (Brihadaranyaka Upanishad 1.4.10)",
        source="Principal Upanishads",
        recommendation="You may ask about moksha or Advaita Vedanta next.",
    ),
    KnowledgeEntry(
        concept="yoga",
        keywords=["yoga", "meditation", "mind", "samadhi", "asana"],
        summary=(
            "In Vedic and Yogic traditions, yoga is disciplined integration of body, mind, and awareness. "
            "Classical paths include karma yoga, bhakti yoga, jnana yoga, and raja yoga."
        ),
        sanskrit_quote="योगश्चित्तवृत्तिनिरोधः (Yoga Sutra 1.2)",
        source="Patanjali Yoga Sutras",
        recommendation="Ask about practical daily routines from yoga philosophy.",
    ),
]

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "hi": {
        "I could not find that in my current Vedic knowledge base. Please ask about concepts like karma, dharma, atman, yoga, or moksha.": "मेरे वर्तमान वैदिक ज्ञान भंडार में इसका उत्तर नहीं मिला। कृपया कर्म, धर्म, आत्मन, योग या मोक्ष जैसे विषयों के बारे में पूछें।",
        "Source": "स्रोत",
        "Recommendation": "सुझाव",
    },
    "sa": {
        "I could not find that in my current Vedic knowledge base. Please ask about concepts like karma, dharma, atman, yoga, or moksha.": "मम वर्तमाने वैदिक-ज्ञान-भाण्डारे एतस्य उत्तरं न लब्धम्। कृपया कर्म, धर्म, आत्मन्, योग, मोक्ष इत्यादिषु पृच्छत।",
        "Source": "स्रोतः",
        "Recommendation": "उपदेशः",
    },
    "es": {
        "I could not find that in my current Vedic knowledge base. Please ask about concepts like karma, dharma, atman, yoga, or moksha.": "No pude encontrar eso en mi base de conocimiento védica actual. Pregunta por karma, dharma, atman, yoga o moksha.",
        "Source": "Fuente",
        "Recommendation": "Recomendación",
    },
}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    language: str = Field(default="en", description="ISO-like short code, e.g., en, hi, sa")
    include_quote: bool = Field(default=True)


class ChatResponse(BaseModel):
    answer: str
    language: str
    concept: Optional[str]
    source: Optional[str]
    recommendation: Optional[str]


def maybe_translate(text: str, language: str) -> str:
    if language == "en":
        return text
    return TRANSLATIONS.get(language, {}).get(text, text)


def find_best_match(user_message: str) -> Optional[KnowledgeEntry]:
    text = user_message.lower()
    best: Optional[KnowledgeEntry] = None
    score = 0
    for entry in KNOWLEDGE_BASE:
        matches = sum(1 for keyword in entry.keywords if keyword in text)
        if matches > score:
            score = matches
            best = entry
    return best if score > 0 else None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return 
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>VedantAI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; margin: 0; background: #f4f1ea; }
      .container { max-width: 880px; margin: 24px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,.08); overflow: hidden; }
      header { background: #2f2a56; color: #fff; padding: 18px; }
      header h1 { margin: 0; font-size: 22px; }
      #chat { height: 480px; overflow-y: auto; padding: 16px; }
      .msg { margin: 10px 0; padding: 10px 12px; border-radius: 10px; line-height: 1.4; }
      .user { background: #e7f0ff; margin-left: 18%; }
      .bot { background: #f5f5f5; margin-right: 18%; }
      .controls { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #ececec; }
      input, select, button { padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
      input { flex: 1; }
      button { background: #2f2a56; color: white; cursor: pointer; }
      .meta { color: #444; font-size: 12px; margin-top: 6px; }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>VedantAI — Multilingual Vedic Chatbot</h1>
      </header>
      <div id="chat"></div>
      <div class="controls">
        <select id="language">
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="sa">Sanskrit</option>
          <option value="es">Spanish</option>
        </select>
        <input id="message" placeholder="Ask about karma, dharma, atman, yoga..." />
        <button onclick="sendMessage()">Send</button>
      </div>
    </div>

    <script>
      const chat = document.getElementById('chat');
      const input = document.getElementById('message');

      function addMsg(text, cls, meta='') {
        const box = document.createElement('div');
        box.className = `msg ${cls}`;
        box.innerHTML = `<div>${text}</div>${meta ? `<div class="meta">${meta}</div>` : ''}`;
        chat.appendChild(box);
        chat.scrollTop = chat.scrollHeight;
      }

      async function sendMessage() {
        const message = input.value.trim();
        if (!message) return;
        const language = document.getElementById('language').value;
        addMsg(message, 'user');
        input.value = '';

        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message, language, include_quote: true})
        });
        const data = await res.json();
        const meta = [
          data.source ? `Source: ${data.source}` : null,
          data.recommendation ? `Recommendation: ${data.recommendation}` : null
        ].filter(Boolean).join(' | ');
        addMsg(data.answer, 'bot', meta);
      }

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
      });

      addMsg('Namaste! I am VedantAI. Ask me Vedic questions in your preferred language.', 'bot');
    </script>
  </body>
</html>



@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    match = find_best_match(req.message)

    if not match:
        fallback = (
            "I could not find that in my current Vedic knowledge base. "
            "Please ask about concepts like karma, dharma, atman, yoga, or moksha."
        )
        return ChatResponse(
            answer=maybe_translate(fallback, req.language),
            language=req.language,
            concept=None,
            source=None,
            recommendation=None,
        )

    answer = match.summary
    if req.include_quote:
        answer += f"\n\nSanskrit reference: {match.sanskrit_quote}"

    source_label = maybe_translate("Source", req.language)
    rec_label = maybe_translate("Recommendation", req.language)

    return ChatResponse(
        answer=answer,
        language=req.language,
        concept=match.concept,
        source=f"{source_label}: {match.source}",
        recommendation=f"{rec_label}: {match.recommendation}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)