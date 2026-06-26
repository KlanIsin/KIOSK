import json
import os
import re
import requests
from rich.console import Console
from rich.markdown import Markdown

MODEL = "cohere/north-mini-code:free"  # Free model, change if desired
URL = "https://openrouter.ai/api/v1/chat/completions"
LOG_FILE = "chat_log.json"  # The file where logs are saved
DOCS_DIR = "docs"  # Local documents used for retrieval
MAX_DOCS = 3
MAX_DOC_SNIPPET = 1500

console = Console()
messages = []

def get_api_key():
  env_key = os.getenv("OPENROUTER_API_KEY")
  if env_key:
      return env_key.strip()

  if os.path.exists("api.txt"):
      with open("api.txt", "r", encoding="utf-8") as f:
          for line in f:
              token = line.strip()
              if token.startswith("sk-"):
                  return token

  raise SystemExit(
      "API key not found. Set OPENROUTER_API_KEY or place a key in api.txt."
  )

API_KEY = get_api_key()


def load_documents(directory):
  documents = []
  if not os.path.isdir(directory):
      return documents

  for name in sorted(os.listdir(directory)):
      if not name.lower().endswith(".txt"):
          continue

      path = os.path.join(directory, name)
      try:
          with open(path, "r", encoding="utf-8") as f:
              text = f.read().strip()
      except Exception:
          continue

      if text:
          documents.append({"name": name, "text": text})

  return documents


def tokenize(text):
  return set(re.findall(r"\w+", text.lower()))


def score_document(query, document):
  query_tokens = tokenize(query)
  doc_tokens = tokenize(document["text"])
  return len(query_tokens & doc_tokens)


def retrieve_documents(query, documents, top_k=MAX_DOCS):
  scored = [
      {"score": score_document(query, doc), "doc": doc}
      for doc in documents
  ]
  scored = [item for item in scored if item["score"] > 0]
  scored.sort(key=lambda item: item["score"], reverse=True)
  return [item["doc"] for item in scored[:top_k]]


def build_rag_context(documents):
  if not documents:
      return None

  snippets = []
  for doc in documents:
      snippet = doc["text"][:MAX_DOC_SNIPPET].strip()
      if len(doc["text"]) > MAX_DOC_SNIPPET:
          snippet += "\n\n[...truncated...]"
      snippets.append(f"### {doc['name']}\n{snippet}")

  return (
      "Use the following reference documents to answer the user's question. "
      "If the answer is not contained in these documents, say you don't know.\n\n"
      + "\n\n".join(snippets)
  )


# Load existing conversation log if it exists.
if os.path.exists(LOG_FILE):
  try:
      with open(LOG_FILE, "r", encoding="utf-8") as f:
          messages = json.load(f)
      console.print(f"[yellow]Loaded {len(messages)} previous messages from log.[/yellow]\n")
  except json.JSONDecodeError:
      console.print("[yellow]Log file was corrupted, starting fresh.[/yellow]\n")
      messages = []


# Load documents for retrieval.
documents = load_documents(DOCS_DIR)
if documents:
  console.print(
      f"[cyan]Loaded {len(documents)} RAG reference document(s) from '{DOCS_DIR}'.[/cyan]\n"
  )
else:
  console.print(
      f"[yellow]No documents found in '{DOCS_DIR}'. Create .txt files there to enable RAG.[/yellow]\n"
  )

console.print("Chatbot started. Type 'quit' to exit.\n", style="bold green")

while True:
  user_input = console.input("[bold blue]You: [/bold blue]")
  if user_input.lower() in ("quit", "exit"):
      break

  messages.append({"role": "user", "content": user_input})

  relevant_docs = retrieve_documents(user_input, documents)
  rag_context = build_rag_context(relevant_docs)

  payload = []
  if rag_context:
      payload.append({"role": "system", "content": rag_context})
  payload.extend(messages)

  response = requests.post(
      URL,
      headers={
          "Authorization": f"Bearer {API_KEY}",
          "Content-Type": "application/json",
      },
      json={
          "model": MODEL,
          "messages": payload,
      },
  )

  if response.status_code == 200:
      reply = response.json()["choices"][0]["message"]["content"]

      console.print("Bot:", style="bold green")
      console.print(Markdown(reply))
      console.print()

      messages.append({"role": "assistant", "content": reply})

      with open(LOG_FILE, "w", encoding="utf-8") as f:
          json.dump(messages, f, indent=2)
  else:
      console.print(f"[bold red]Error: {response.status_code} - {response.text}[/bold red]\n")
