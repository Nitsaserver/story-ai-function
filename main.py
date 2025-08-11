import functions_framework
import google.generativeai as genai
import os
import json
from google.cloud import firestore
from google.cloud.firestore_v1.base_document import DocumentSnapshot

# Read the Gemini API Key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

# Configure the Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# Firestore client
db = firestore.Client()

# CORS settings
ALLOWED_ORIGINS = ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:5175']


def get_user_preferences(user_id: str, app_id: str) -> dict:
    preferences_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/preferences').document('story_prefs')
    preferences_doc: DocumentSnapshot = preferences_ref.get()
    if preferences_doc.exists:
        return preferences_doc.to_dict() or {}
    return {}

def update_user_preferences(user_id: str, app_id: str, feedback_type: str):
    preferences_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/preferences').document('story_prefs')
    preferences_ref.set({
        'last_feedback_type': feedback_type,
        'updated_at': firestore.SERVER_TIMESTAMP
    }, merge=True)


@functions_framework.http
def generate_story_function(request):
    # Determine the origin of the request
    request_origin = request.headers.get('Origin')

    # Set CORS headers
    headers = {
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }

    if request_origin and (request_origin in ALLOWED_ORIGINS or '*' in ALLOWED_ORIGINS):
        headers['Access-Control-Allow-Origin'] = request_origin
    else:
        headers['Access-Control-Allow-Origin'] = '*'

    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    # Parse JSON request body
    request_json = request.get_json(silent=True)
    if not request_json:
        return (json.dumps({'error': 'Invalid JSON or missing request body'}), 400, headers)

    keywords = request_json.get('keywords', '').strip()
    user_id = request_json.get('userId')
    app_id = request_json.get('appId')

    if not keywords:
        return (json.dumps({'error': 'Missing keywords for story generation'}), 400, headers)
    if not user_id:
        return (json.dumps({'error': 'Missing userId'}), 400, headers)
    if not app_id:
        return (json.dumps({'error': 'Missing appId'}), 400, headers)

    try:
        # Build prompt for Gemini
        prompt_parts = [
            f"Generate a short, engaging story based on these keywords: {keywords}.",
            "Keep it concise, around 100-150 words."
        ]
        full_prompt = " ".join(prompt_parts)

        # Create the Gemini model instance
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Generate the story
        response = model.generate_content(full_prompt)
        generated_story = response.text

        # Optionally update Firestore with feedback (not used in this example)
        update_user_preferences(user_id, app_id, feedback_type="generated")

        return (json.dumps({'story': generated_story}), 200, headers)

    except Exception as e:
        print(f"Error generating story: {e}")
        return (json.dumps({'error': str(e)}), 500, headers)