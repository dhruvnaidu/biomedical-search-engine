import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from .config import GEMINI_MODEL_NAME

# Load env variables
load_dotenv()

class GeminiReranker:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def rerank(self, query, candidates):
        print("🧠 Re-ranking with Gemini...")
        
        # Prepare candidates for prompt
        candidate_list = [
            {"title": c["title"], "text": c["text"]} for c in candidates
        ]

        system_prompt = """
        You are an expert biomedical information retrieval and NLP researcher.

        Your task: Given a user query and a list of biomedical paper titles or abstracts, rank the papers by their relevance to the query.

        Follow these guidelines:
        - Use domain knowledge and reasoning, not keyword overlap.
        - Consider clinical, molecular, and methodological alignment.
        - Prefer papers that directly address the biological mechanism, disease, or intervention described in the query.
        - Be concise but accurate in summaries.
        - Output valid JSON only (no extra text).

        Return the result in the following JSON format:

        {
        "query_understanding": "<succinct restatement of the biomedical question or topic in your own words>",
        "ranking_method": "Semantic and biomedical relevance based on mechanistic, clinical, or methodological similarity.",
        "ranked_results": [
            {
            "title": "<paper title>",
            "relevance_score": <float between 0 and 1>,
            "relevance_reason": "<1–2 sentence biomedical reasoning>",
            "summary": "<2–3 sentence summary of the paper’s biomedical focus>"
            }
        ]
        }
    """
        
        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\nCandidates:\n{json.dumps(candidate_list)}"
            )
            
            # Clean response (remove markdown code blocks if present)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"⚠️ Error during LLM Reranking: {e}")
            return {"ranked_results": []}