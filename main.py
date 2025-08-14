# main.py

import functions_framework
import google.generativeai as genai
import os
import json
from google.cloud import firestore
from datetime import datetime
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1._helpers import DatetimeWithNanoseconds



# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

print("[Gemini Config] Using API key authentication")
print(f"[Gemini Config] GEMINI_API_KEY set: {bool(GEMINI_API_KEY)}")

try:
    models_list = list(genai.list_models())
    print(f"[Gemini Config] API key valid. {len(models_list)} models available.")
except Exception as e:
    print(f"[Gemini Config] ERROR: API key check failed — {e}")

db = firestore.Client()

ALLOWED_ORIGINS = [
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:5175'
]

def safe_json_dumps(data):
    def default(o):
        if isinstance(o, DatetimeWithNanoseconds):
            return o.isoformat()
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)  # fallback for other non-serializable types
    return json.dumps(data, default=default)

# --- Firestore helpers ---
def get_user_preferences(user_id: str, app_id: str) -> dict:
    doc = db.collection(
        f'artifacts/{app_id}/users/{user_id}/preferences'
    ).document('story_prefs').get()
    return doc.to_dict() if doc.exists else {}

def update_user_preferences(user_id: str, app_id: str, update_data: dict):
    ref = db.collection(
        f'artifacts/{app_id}/users/{user_id}/preferences'
    ).document('story_prefs')
    ref.set({**update_data, 'last_updated': firestore.SERVER_TIMESTAMP}, merge=True)

# --- Agentic planning step ---
def agent_plan(user_id: str, app_id: str, keywords: str) -> dict:
    prefs = get_user_preferences(user_id, app_id)
    planner_prompt = f"""
    You are a creative story planning agent.
    User preferences: {safe_json_dumps(prefs)}
    Keywords: {keywords}

    Based on these, plan the best story approach.
    Return JSON with:
    - tone: the tone to use ("humorous", "adventurous", "positive", "neutral", etc.)
    - plot_outline: 2-4 sentences describing the story's main arc
    - length_in_words: integer (between 100 and 200)
    """

    planner_model = genai.GenerativeModel('gemini-1.5-pro')
    plan_response = planner_model.generate_content(planner_prompt)
    
    try:
        plan = json.loads(plan_response.text)
    except json.JSONDecodeError:
        # fallback if model returns plain text
        plan = {
            "tone": prefs.get("preferred_tone", "neutral"),
            "plot_outline": f"Story based on: {keywords}",
            "length_in_words": 150
        }
    return plan

# --- Story execution step ---
def generate_story_from_plan(plan: dict) -> str:
    execution_prompt = f"""
    Write a {plan['length_in_words']}-word story.
    Tone: {plan['tone']}
    Plot outline: {plan['plot_outline']}
    Ensure it's engaging, coherent, and tailored to the tone.
    """

    executor_model = genai.GenerativeModel('gemini-1.5-flash')
    result = executor_model.generate_content(execution_prompt)
    return result.text

# --- Cloud Function ---
@functions_framework.http
def generate_story_function(request):
    request_origin = request.headers.get('Origin')
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, X-Requested-With, Authorization',
        'Access-Control-Max-Age': '3600'
    }

    if request_origin and (request_origin in ALLOWED_ORIGINS or '*' in ALLOWED_ORIGINS):
        headers['Access-Control-Allow-Origin'] = request_origin

    if request.method == 'OPTIONS':
        return ('', 204, headers)

    request_json = request.get_json(silent=True)
    if not request_json:
        return (safe_json_dumps({'error': 'Invalid JSON or missing request body'}), 400, headers)

    keywords = request_json.get('keywords', '').strip()
    user_id = request_json.get('userId')
    app_id = request_json.get('appId')
    feedback_type = request_json.get('feedbackType')

    if not user_id:
        return (safe_json_dumps({'error': 'Missing userId'}), 400, headers)
    if not app_id:
        return (safe_json_dumps({'error': 'Missing appId'}), 400, headers)

    try:
        # Handle feedback updates only
        if feedback_type:
            prefs = get_user_preferences(user_id, app_id)
            feedback_counts = prefs.get('feedback_counts', {})
            feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1
            update_user_preferences(user_id, app_id, {"feedback_counts": feedback_counts})
            return (safe_json_dumps({'message': f'Feedback {feedback_type} processed.'}), 200, headers)

        if not keywords:
            return (safe_json_dumps({'error': 'Missing keywords for story generation'}), 400, headers)

        # Phase 1: Plan
        plan = agent_plan(user_id, app_id, keywords)

        # Phase 2: Generate story
        story = generate_story_from_plan(plan)

        # Phase 3: Memory update (store tone, last plan, last keywords)
        update_user_preferences(user_id, app_id, {
            "preferred_tone": plan.get("tone", "neutral"),
            "last_keywords": keywords,
            "last_plan": plan
        })

        return (safe_json_dumps({'plan': plan, 'story': story}), 200, headers)

    except Exception as e:
        print(f"Error: {e}")
        return (safe_json_dumps({'error': str(e)}), 500, headers)
