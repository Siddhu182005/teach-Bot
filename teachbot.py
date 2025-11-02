# tutorbot.py
# Simple TutorBot: Flask server + minimal frontend.
# Usage:
# 1) export OPENAI_API_KEY="sk-..."
# 2) python tutorbot.py
# 3) Open http://127.0.0.1:5000 in browser

from flask import Flask, request, jsonify, render_template_string
import os, openai, numpy as np, math, time

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")

app = Flask(__name__)

# In-memory storage for documents:
# stored_docs: list of dicts: {"id":int, "text": str, "chunks": [str], "embs": [np.array]}
stored_docs = []
doc_counter = 0

# --- Utilities ---
def chunk_text(text, max_words=200, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
        i += max_words - overlap
    return chunks

def embed_texts(texts, model="text-embedding-3-small"):
    # call OpenAI embeddings in batch-friendly way
    # returns list of numpy arrays
    resp = openai.Embeddings.create(model=model, input=texts)
    embs = [np.array(item["embedding"], dtype=np.float32) for item in resp["data"]]
    return embs

def cosine_sim(a, b):
    # prevent div by zero
    if a is None or b is None: return -1.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return -1.0
    return float(np.dot(a, b) / (an * bn))

def retrieve_top_k(query, k=4):
    # returns top k (chunk, score, doc_id)
    if not stored_docs:
        return []
    q_emb = embed_texts([query])[0]
    candidates = []
    for doc in stored_docs:
        for i, chunk_emb in enumerate(doc["embs"]):
            score = cosine_sim(q_emb, chunk_emb)
            candidates.append({"chunk": doc["chunks"][i], "score": score, "doc_id": doc["id"]})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]

# --- System prompt template ---
SYSTEM_PROMPT = """You are TutorBot — a clear, patient tutor. Use ONLY the SOURCE CONTENT provided below to answer the student's question whenever possible.
Rules:
1) First give a short explanation in the student's chosen style. Keep it simple and stepwise.
2) Then give 1 short example (or illustration).
3) Then give 2 short quiz questions (one multiple-choice or short-answer).
4) If you cannot find the answer in the SOURCE CONTENT, say "I don't see that in the provided material." Then offer a labeled best-effort answer.
Always be humble about uncertainty. Keep language beginner-friendly.
"""

# --- Routes ---
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/ingest", methods=["POST"])
def ingest():
    global doc_counter
    payload = request.json or {}
    text = payload.get("text", "").strip()
    title = payload.get("title", f"doc_{int(time.time())}")
    if not text:
        return jsonify({"ok": False, "error": "No text provided."}), 400

    chunks = chunk_text(text)
    embs = embed_texts(chunks)
    doc = {"id": doc_counter, "title": title, "text": text, "chunks": chunks, "embs": embs}
    stored_docs.append(doc)
    doc_counter += 1
    return jsonify({"ok": True, "doc_id": doc["id"], "num_chunks": len(chunks)})

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.json or {}
    question = payload.get("question", "").strip()
    style = payload.get("style", "Concise")
    if not question:
        return jsonify({"ok": False, "error": "No question provided."}), 400

    # retrieve relevant chunks
    retrieved = retrieve_top_k(question, k=5)
    if retrieved:
        source_text = "\n\n--- Retrieved Chunks (most relevant first) ---\n\n"
        for r in retrieved:
            source_text += f"[score {r['score']:.3f}] {r['chunk']}\n\n"
    else:
        source_text = "No documents ingested."

    # compose prompt to LLM
    user_instructions = f"""
STUDENT PREFERRED STYLE: {style}
QUESTION: {question}

Use the SOURCE below to answer. Follow the TutorBot rules in system prompt.
SOURCE:
{source_text}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instructions}
    ]

    # call OpenAI chat completion (choose a chat-capable model)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",   # change if you prefer another model
            messages=messages,
            max_tokens=600,
            temperature=0.2
        )
        answer = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"ok": False, "error": f"OpenAI API error: {e}"}), 500

    return jsonify({"ok": True, "answer": answer, "retrieved": retrieved})

# --- Minimal frontend (keeps it tiny) ---
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TutorBot — simple</title>
  <style>
    body { font-family: system-ui, Arial; max-width:900px; margin:20px auto; padding:10px; }
    textarea { width:100%; height:120px; font-size:14px; }
    input, button, select { font-size:14px; padding:6px; }
    .chat { border:1px solid #ddd; padding:10px; height:360px; overflow:auto; background:#fafafa; }
    .me { color: white; background:#0b84ff; padding:8px; border-radius:6px; display:inline-block; margin:6px 0; }
    .bot { color:#111; background:#e8e8e8; padding:8px; border-radius:6px; display:inline-block; margin:6px 0; }
    .row { display:flex; gap:8px; margin-top:8px; }
  </style>
</head>
<body>
  <h2>TutorBot — paste material, then ask</h2>

  <label><strong>Paste document / notes:</strong></label>
  <textarea id="docText" placeholder="Paste lecture notes, article, or any text here..."></textarea>
  <div class="row">
    <input id="docTitle" placeholder="Document title (optional)" />
    <button onclick="ingest()">Ingest document</button>
    <span id="ingestStatus"></span>
  </div>

  <hr/>

  <div class="chat" id="chatWindow"></div>

  <div style="margin-top:10px;">
    <select id="styleSelect">
      <option>Concise</option>
      <option>Analogy-first</option>
      <option>Step-by-step</option>
      <option>Socratic</option>
    </select>
    <input id="questionInput" style="width:70%" placeholder="Ask a question about the pasted material..." />
    <button onclick="ask()">Ask</button>
  </div>

  <script>
    async function ingest(){
      const text = document.getElementById("docText").value;
      const title = document.getElementById("docTitle").value;
      if(!text.trim()){ alert("Paste something first."); return; }
      document.getElementById("ingestStatus").innerText = "…ingesting";
      const res = await fetch("/ingest", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text: text, title: title})
      });
      const j = await res.json();
      if(j.ok){
        document.getElementById("ingestStatus").innerText = "Ingested — chunks: " + j.num_chunks;
        appendChat("bot", "Document ingested. You can now ask questions about it.");
      } else {
        document.getElementById("ingestStatus").innerText = "Error";
        appendChat("bot", "Ingest error: " + (j.error || "unknown"));
      }
    }

    function appendChat(who, text){
      const c = document.getElementById("chatWindow");
      const el = document.createElement("div");
      el.className = who==="me" ? "me" : "bot";
      el.innerText = text;
      c.appendChild(el);
      c.scrollTop = c.scrollHeight;
    }

    async function ask(){
      const q = document.getElementById("questionInput").value;
      const style = document.getElementById("styleSelect").value;
      if(!q.trim()){ alert("Ask something."); return; }
      appendChat("me", q);
      appendChat("bot", "…thinking (calling tutor)");
      document.getElementById("questionInput").value = "";
      const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({question: q, style: style})
      });
      const j = await res.json();
      // remove "…thinking" last child
      const chat = document.getElementById("chatWindow");
      if(chat.lastChild && chat.lastChild.innerText.includes("…thinking")) chat.removeChild(chat.lastChild);

      if(j.ok){
        appendChat("bot", j.answer);
      } else {
        appendChat("bot", "Error: " + (j.error || "unknown"));
      }
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
