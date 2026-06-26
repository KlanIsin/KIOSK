# School AI Helpdesk

This project is a Flask-based school AI helpdesk with local RAG support and a static HTML frontend.

## Features
- Flask backend with `/api/chat` endpoint
- Static HTML/JavaScript frontend at `/`
- Local school documents in `docs/` used for retrieval-augmented generation
- OpenRouter API integration for model responses

## Setup
1. Install dependencies in your Python environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your API key in the environment or `api.txt`:
   - `OPENROUTER_API_KEY` environment variable
   - or place a key in `api.txt`
   - or create a `.env` file with the key and other settings
3. Add school documents to the `docs/` folder.
   - You may now place text-based PDF files in `docs/` as well as `.txt` files.
   - Scanned PDFs are not supported yet and would require OCR.

## Run
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Add school context
Create `.txt` files inside `docs/` with policies, schedules, contact info, student services, and other school-specific details.

## Notes
- The helpdesk uses simple keyword overlap retrieval. For better accuracy, add focused school docs.
- If you want support for document uploads or persistent sessions, I can extend the app further.
