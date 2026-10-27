# Web apps, LLMs and RAG

## Flask for data scientists
Flask turns a model into something other people can use:
```python
app = Flask(__name__)
model = joblib.load("models/housing_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    df = pd.DataFrame([request.get_json()])
    return jsonify(price=float(model.predict(df)[0]))
```
- Routes, templates (`render_template` + Jinja), static files (`/static/`) and forms.
- Return JSON for APIs. Load the model once, at start-up, not on every request.

## Large language models
- **Tokens and context window:** LLMs read and write tokens and can only "see" a limited window at once.
- **Prompting:** give a role, the task, context, the output format and examples (few-shot). Lower temperature gives more deterministic answers.
- **Limits:** knowledge stops at a cut-off date, LLMs can hallucinate, and they don't know your private data.

## Retrieval-Augmented Generation (RAG)
RAG fixes those limits by fetching relevant documents and putting them in the prompt:

```
documents → chunks (+ metadata) → embeddings → vector index
question  → embedding → top-k similar chunks → prompt(question + chunks) → LLM → grounded answer + sources
```
- **Embeddings** map text to vectors whose cosine similarity reflects meaning.
- **Chunk size** is a trade-off: small chunks are precise but lose context, while big chunks keep context but dilute relevance.
- Better retrieval comes from hybrid search (keywords + embeddings), re-ranking, and metadata filters.

**Project 3** (RAG teaching assistant for course videos: Whisper → chunks → embeddings → LLM) is being rebuilt and will be added to this repo.
