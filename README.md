# ConsensusPrompt ⚖️

**ConsensusPrompt** is an advanced multi-agent prompt optimisation middleware. It uses a **Peer-Review Council Protocol** powered strictly by the lightweight **Gemma 3 1B** model (`gemma-3-1b-it`) via the Google Gemini API. 

Instead of relying on a single static LLM to rewrite your prompt, ConsensusPrompt instantiates a miniature society of AI agents that independently draft, review, and evaluate prompt variations to synthesise the objectively superior result.

## 🌟 The Core Pipeline

1. **Intent Extraction:** The user's initial query is parsed by an extraction agent to identify the domain, the central intent, and implicit missing constraints.
2. **Parallel Generation:** Three independent Rewriter Agents generate diverse perspectives:
   - **Candidate A:** Chain-of-Thought Reasoning
   - **Candidate B:** Role-Assignment & Few-Shot Formatting
   - **Candidate C:** Structured Domain Templates
3. **Anonymised Council Review:** A multi-agent council evaluates the anonymised candidates simultaneously on Clarity, Completeness, Faithfulness, and Domain Fit.
4. **Chairman Arbitration:** The Chairman Agent ingests all peer-review ballots and finalises an aggregate ranking, electing to either forward the winning prompt or intelligently synthesise a hybrid configuration. 
5. **Execution:** The polished, bulletproof prompt is executed safely on Gemma.

## 🚀 Features

- **Dynamic Flowchart Visualization:** The frontend features a beautiful 2D data-flow diagram mapped using animated SVG bezier curves to visually explain the hidden council review operation to the end user.
- **Full Theme Support:** Fully styled Light/Dark mode UI components.
- **Local Analytics Storage:** Built-in persistence layer captures and parses user rating metrics (quality, trust, control) into `backend/feedback.json`.
- **100% Gemma 3:** Built off the absolute latest `langchain-google-genai` integration with zero reliance on OpenAI/Anthropic keys. 

---

## 🛠️ Installation & Setup

### 1. Backend (FastAPI + Python)

```bash
cd consensusprompt/backend

# Create a virtual environment and install packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -U langchain-google-genai google-generativeai

# Configure Google AI Studio Key
echo "GOOGLE_API_KEY=your_gemini_api_key" > .env

# Run the API Server
uvicorn main:app --port 8000 --reload
```

### 2. Frontend (Next.js + React)

```bash
cd consensusprompt/frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the interface.

---

### Project Architecture
- `backend/agents/`: LLM definitions defining the Intent, Arbitration, Council, and Draft logic.
- `backend/pipeline/graph.py`: The orchestrator determining agent state transitions.
- `frontend/app/CouncilScene.tsx`: The animated 2D SVG flowchart component. 
- `frontend/app/page.tsx`: The primary Next.js page controller handling API hooks.
