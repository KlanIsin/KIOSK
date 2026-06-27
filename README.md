# School AI Helpdesk

A simple Flask app that answers school questions using a local knowledge base.

## What it does

- Accepts questions in English and Cantonese
- Uses local `.txt` and `.pdf` files as school knowledge
- Returns answers formatted in Markdown
- Supports document upload through the web UI

## Features

- `/api/chat` for user questions
- `/api/suggestions` for suggested prompts
- `/api/upload` for adding documents
- Static frontend available at `/`
- English and Cantonese language support
- Local document retrieval with optional embedding acceleration

## Requirements

- Python 3.10 or newer
- Flask
- requests
- PyPDF2
- python-dotenv
- sentence-transformers

Optional:

- `faiss-cpu` for faster embedding search

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Install FAISS for faster search:

```bash
pip install faiss-cpu
```

3. Configure your OpenRouter API key:

- Set `OPENROUTER_API_KEY` as an environment variable
- or add a `.env` file with:

```env
OPENROUTER_API_KEY=your_key_here
```

4. Put school documents in `knowledge_base/`:

- `.txt` files
- text-based `.pdf` files

> Scanned PDFs are not supported without OCR.

## Run

From the project root, start the app with:

```bash
python src/app.py
```

Or, equivalently:

```bash
python -m src.app
```

Then open:

```
http://127.0.0.1:5000
```

## Add school content

Create files in `knowledge_base/` containing:

- school policies
- schedules
- contact information
- student services
- other school-related details

## Notes

- Uploaded documents are saved to `knowledge_base/`
- The app works without `faiss`, but search is faster with it
