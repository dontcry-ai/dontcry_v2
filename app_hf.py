"""
Flask API for Baby Cry Classification using Hugging Face API
Memory optimized - uses ~150MB RAM instead of 1.5GB
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import requests
import librosa
from datetime import datetime
import base64
from pydub import AudioSegment
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hugging Face Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/dontcryai/dontcry"
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Cry categories mapping
CRY_CATEGORIES = {
    "LABEL_0": "hungry",
    "LABEL_1": "tired",
    "LABEL_2": "belly_pain",
    "LABEL_3": "burping",
    "LABEL_4": "discomfort"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_to_wav_bytes(file_path):
    """Convert audio to WAV format in memory"""
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        raise Exception(f"Audio conversion failed: {e}")


def preprocess_audio(audio_path):
    """Load and preprocess audio to 16kHz WAV"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Normalize
        if audio.max() > 0:
            audio = audio / audio.max()
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Ensure 5 seconds duration
        target_length = 16000 * 5
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = librosa.util.pad_center(audio, size=target_length)
        
        # Convert to bytes
        import soundfile as sf
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, 16000, format='WAV')
        wav_io.seek(0)
        
        return wav_io.read()
    except Exception as e:
        raise Exception(f"Audio preprocessing failed: {e}")


def call_huggingface_api(audio_bytes, max_retries=3):
    """Call Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 503:
                # Model is loading
                if attempt < max_retries - 1:
                    print(f"Model loading, retry {attempt + 1}/{max_retries}...")
                    import time
                    time.sleep(5)  # Wait 5 seconds
                    continue
                else:
                    return {"error": "Model is loading, please try again in a moment"}
            
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            return {"error": "Request timeout"}
        
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": "Max retries reached"}


def format_prediction_result(hf_response, confidence_threshold=0.6):
    """Format HF API response to match your frontend expectations"""
    try:
        if isinstance(hf_response, dict) and "error" in hf_response:
            return None, hf_response["error"]
        
        # HF returns: [{'label': 'LABEL_0', 'score': 0.85}, ...]
        # Convert to your format
        
        # Get top prediction
        top_pred = hf_response[0]
        predicted_label = CRY_CATEGORIES.get(top_pred['label'], 'unknown')
        confidence = top_pred['score']
        
        # Build all probabilities
        all_probs = {}
        for item in hf_response:
            category = CRY_CATEGORIES.get(item['label'], item['label'])
            all_probs[category] = item['score']
        
        result = {
            'predicted_class': predicted_label,
            'confidence': float(confidence),
            'all_probabilities': all_probs,
            'meets_threshold': confidence >= confidence_threshold
        }
        
        return result, None
    
    except Exception as e:
        return None, f"Error formatting result: {e}"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        "message": "DontCryAI Backend API (Hugging Face)",
        "status": "running",
        "version": "3.0-HF",
        "model": "dontcryai/baby-cry-classifier",
        "endpoints": {
            "health": "/api/health",
            "predict_upload": "/api/predict/upload",
            "predict_record": "/api/predict/record",
            "get_classes": "/api/classes"
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    # Test HF API connection
    hf_status = "connected" if HF_TOKEN else "no_token"
    
    return jsonify({
        'status': 'healthy',
        'hugging_face': hf_status,
        'model': 'dontcryai/baby-cry-classifier',
        'classes': list(CRY_CATEGORIES.values()),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Predict from uploaded file"""
    try:
        if not HF_TOKEN:
            return jsonify({
                'success': False,
                'error': 'Hugging Face token not configured'
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file format'
            }), 400
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        
        # Save temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        try:
            # Preprocess audio
            audio_bytes = preprocess_audio(temp_path)
            
            # Call Hugging Face API
            hf_response = call_huggingface_api(audio_bytes)
            
            # Format result
            prediction_result, error = format_prediction_result(hf_response, confidence_threshold)
            
            if error:
                return jsonify({
                    'success': False,
                    'error': error
                }), 500
            
            return jsonify({
                'success': True,
                'data': {
                    'prediction': prediction_result,
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Error in predict_upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/record', methods=['POST'])
def predict_record():
    """Predict from recording"""
    try:
        if not HF_TOKEN:
            return jsonify({
                'success': False,
                'error': 'Hugging Face token not configured'
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        audio_format = data.get('format', 'base64')
        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        
        # Decode audio
        if audio_format == 'base64':
            audio_base64 = data.get('audio_data')
            if not audio_base64:
                return jsonify({
                    'success': False,
                    'error': 'No audio_data provided'
                }), 400
            
            try:
                audio_bytes = base64.b64decode(audio_base64)
                temp_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    f'temp_record_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
                )
                
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Preprocess
                processed_audio = preprocess_audio(temp_path)
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Audio decode error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode audio data'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Only base64 format supported'
            }), 400
        
        # Call Hugging Face API
        hf_response = call_huggingface_api(processed_audio)
        
        # Format result
        prediction_result, error = format_prediction_result(hf_response, confidence_threshold)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': prediction_result,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"Error in predict_record: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get cry types"""
    return jsonify({
        'success': True,
        'classes': list(CRY_CATEGORIES.values()),
        'num_classes': len(CRY_CATEGORIES)
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 10MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# STARTUP
# ============================================================================

print("=" * 70)
print("DONTCRYAI BACKEND - HUGGING FACE API MODE")
print("=" * 70)
print(f"✓ Model: dontcryai/baby-cry-classifier")
print(f"✓ HF Token: {'Configured' if HF_TOKEN else 'Missing'}")
print(f"✓ Categories: {list(CRY_CATEGORIES.values())}")
print(f"✓ Memory: ~150MB (vs 1.5GB local)")
print("=" * 70)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server on http://0.0.0.0:{port}\n")

    app.run(host='0.0.0.0', port=port, debug=False)

