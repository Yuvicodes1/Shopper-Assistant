from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
from collections import defaultdict
import json
import random
import urllib.parse

# =============== Load NLP + LLM =================
print("Loading spaCy...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download if not present
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("Loading FLAN-T5...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print("Loading Whisper ASR...")
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# =============== Enhanced Product Knowledge Base ================
PRODUCT_CATEGORIES = {
    "footwear": {
        "keywords": ["shoes", "sneakers", "boots", "sandals", "slippers", "heels", "flats", "loafers", "running shoes", "sports shoes"],
        "brands": ["Nike", "Adidas", "Puma", "Reebok", "Vans", "Converse", "New Balance", "Asics"],
        "features": ["size", "color", "material", "sole", "cushioning", "breathable", "waterproof"]
    },
    "clothing": {
        "keywords": ["shirt", "t-shirt", "tshirt", "jeans", "pants", "dress", "jacket", "hoodie", "sweater", "shorts", "top", "blouse"],
        "brands": ["Zara", "H&M", "Uniqlo", "Levi's", "Nike", "Adidas", "Gap"],
        "features": ["size", "color", "material", "fit", "sleeve", "collar", "pattern"]
    },
    "electronics": {
        "keywords": ["phone", "laptop", "tablet", "headphones", "speaker", "charger", "cable", "mouse", "keyboard"],
        "brands": ["Apple", "Samsung", "Sony", "Dell", "HP", "Lenovo", "Xiaomi", "OnePlus"],
        "features": ["brand", "model", "storage", "ram", "battery", "display", "camera", "price"]
    },
    "accessories": {
        "keywords": ["bag", "backpack", "wallet", "watch", "belt", "sunglasses", "jewelry", "hat"],
        "brands": ["Ray-Ban", "Fossil", "Michael Kors", "Coach", "Prada", "Gucci"],
        "features": ["material", "color", "size", "brand", "style", "waterproof"]
    }
}

# =============== Enhanced Price Recognition Patterns ================
PRICE_PATTERNS = {
    # USD patterns
    "usd": [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $50, $1,000, $99.99
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd|USD)',  # 50 dollars, 100 USD
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:bucks?)',  # 50 bucks
    ],
    # INR patterns
    "inr": [
        r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # ₹1000, ₹1,50,000
        r'(?:rs\.?|rupees?|inr|INR)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Rs. 1000, INR 5000
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rupees?|rs\.?|inr|INR)',  # 1000 rupees, 5000 INR
    ],
    # EUR patterns
    "eur": [
        r'€\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # €50, €1,000
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:euros?|eur|EUR)',  # 50 euros, 100 EUR
    ],
    # GBP patterns
    "gbp": [
        r'£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # £50, £1,000
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:pounds?|gbp|GBP)',  # 50 pounds, 100 GBP
    ],
    # Generic number patterns (when currency not specified)
    "generic": [
        r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\b',  # Any number format
    ]
}

# Price range keywords
PRICE_RANGE_KEYWORDS = {
    "under": ["under", "below", "less than", "max", "maximum", "up to", "within", "not more than"],
    "over": ["over", "above", "more than", "min", "minimum", "at least", "starting from"],
    "around": ["around", "about", "approximately", "roughly", "near", "close to"],
    "between": ["between", "from", "to", "range"]
}

# =============== Enhanced Feature Patterns ================
FEATURE_PATTERNS = {
    "size": r"\b(?:size|sized?)\s*(\d+(?:\.\d)?|\w+)\b|(\d+(?:\.\d)?)\s*(?:inch|cm|mm|xl|lg|md|sm|xs)\b|\b(xl|lg|md|sm|xs|xxl|xxxl|2xl|3xl)\b",
    "color": r"\b(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|silver|gold|navy|maroon|violet|cyan|magenta|beige|tan|khaki|crimson|emerald|sapphire|ruby|pearl|coral|turquoise|lavender|mint|rose|wine|cream|ivory|copper|bronze|platinum)\b",
    "material": r"\b(cotton|leather|denim|polyester|wool|silk|canvas|plastic|metal|wood|glass|nylon|spandex|lycra|linen|cashmere|velvet|suede|rubber|ceramic|bamboo|organic|synthetic)\b",
    "brand": r"\b(nike|adidas|puma|apple|samsung|sony|zara|h&m|uniqlo|levi's|reebok|vans|converse|new balance|asics|ray-ban|fossil|michael kors|coach|prada|gucci|dell|hp|lenovo|xiaomi|oneplus)\b",
    "style": r"\b(casual|formal|sporty|elegant|vintage|modern|classic|trendy|slim|regular|loose|tight|fitted|straight|skinny|relaxed|comfort|athletic|business|smart|chic|bohemian|minimalist)\b",
    "occasion": r"\b(work|office|gym|running|party|wedding|casual|daily|travel|workout|exercise|formal|business|vacation|date|night|outdoor|indoor|summer|winter|spring|autumn|fall)\b"
}

# =============== Human-like Response Templates ================
RESPONSE_TEMPLATES = {
    "greeting": [
        "Hi there! I'd be happy to help you find what you're looking for.",
        "Hello! I'm here to help you find the perfect products.",
        "Hey! I'm excited to help you find exactly what you need.",
        "Welcome! I'm here to help you discover some great options."
    ],
    "product_found": [
        "Great choice! I found some excellent {product} options for you.",
        "Perfect! I've found some amazing {product} that match what you're looking for.",
        "Awesome! I've got some fantastic {product} recommendations for you.",
        "Wonderful! I found some great {product} options that should work perfectly."
    ],
    "brand_specific": [
        "Excellent taste! {brand} has some amazing {product} options.",
        "Great choice! {brand} makes some fantastic {product}.",
        "Perfect! {brand} has some really good {product} options.",
        "Nice! {brand} has some excellent {product} that I think you'll love."
    ],
    "price_range": [
        "I found some great options {price_range}.",
        "Perfect! There are some excellent choices {price_range}.",
        "Good news! I found some amazing options {price_range}.",
        "Great! I've found some wonderful products {price_range}."
    ],
    "refinement": [
        "Let me narrow that down for you.",
        "I'll help you find exactly what you need.",
        "Let me find the perfect match for you.",
        "I'll help you find something more specific."
    ],
    "multiple_options": [
        "I found several great options for you to choose from.",
        "There are some excellent choices available.",
        "I've found multiple great options that should work well.",
        "Here are some fantastic options to consider."
    ]
}

# =============== Session Management ================
class ConversationSession:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.conversation_history = []
        self.current_search_context = {
            "products": [],
            "features": defaultdict(list),
            "context": [],
            "intent": "browse"
        }
    
    def update_context(self, new_info):
        """Update the current search context with new information"""
        # Merge products
        for product in new_info.get("products", []):
            if product not in self.current_search_context["products"]:
                self.current_search_context["products"].append(product)
        
        # Merge features
        for feature_type, values in new_info.get("features", {}).items():
            for value in values:
                if value not in self.current_search_context["features"][feature_type]:
                    self.current_search_context["features"][feature_type].append(value)
        
        # Merge context
        for ctx in new_info.get("context", []):
            if ctx not in self.current_search_context["context"]:
                self.current_search_context["context"].append(ctx)
        
        # Update intent
        if new_info.get("intent"):
            self.current_search_context["intent"] = new_info["intent"]
    
    def add_message(self, user_msg, assistant_msg):
        self.conversation_history.append(f"User: {user_msg}")
        self.conversation_history.append(f"Assistant: {assistant_msg}")
        # Keep only last 10 messages
        self.conversation_history = self.conversation_history[-10:]

# Global session instance
session = ConversationSession()

# =============== Enhanced Price Extraction ================
def extract_price_info(text):
    """Extract comprehensive price information from text"""
    price_info = []
    text_lower = text.lower()
    
    # Check for price range keywords first
    for range_type, keywords in PRICE_RANGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Now look for prices near this keyword
                for currency, patterns in PRICE_PATTERNS.items():
                    for pattern in patterns:
                        # Search around the keyword
                        keyword_pos = text_lower.find(keyword)
                        search_text = text[max(0, keyword_pos-20):keyword_pos+50]
                        
                        matches = re.findall(pattern, search_text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                amount = next((g for g in match if g), "")
                            else:
                                amount = match
                            
                            if amount:
                                price_info.append({
                                    "amount": amount,
                                    "currency": currency,
                                    "range_type": range_type,
                                    "original_text": f"{keyword} {amount}"
                                })
    
    # If no range keywords found, search for any price mentions
    if not price_info:
        for currency, patterns in PRICE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        amount = next((g for g in match if g), "")
                    else:
                        amount = match
                    
                    if amount:
                        price_info.append({
                            "amount": amount,
                            "currency": currency,
                            "range_type": "exact",
                            "original_text": match
                        })
    
    return price_info

# =============== Enhanced Processing Functions ================
def extract_product_info(text, previous_context=None):
    """Extract comprehensive product information from user text"""
    doc = nlp(text.lower())
    
    # Initialize result structure
    result = {
        "products": [],
        "features": defaultdict(list),
        "context": [],
        "intent": "browse"
    }
    
    # 1. Extract products using multiple methods
    products = set()
    
    # Method 1: Direct keyword matching
    for category, info in PRODUCT_CATEGORIES.items():
        for keyword in info["keywords"]:
            if keyword in text.lower():
                products.add(keyword)
    
    # Method 2: NER entities
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "PERSON"]:
            ent_lower = ent.text.lower()
            for category, info in PRODUCT_CATEGORIES.items():
                if ent_lower in [b.lower() for b in info["brands"]]:
                    result["features"]["brand"].append(ent.text.title())
                elif any(keyword in ent_lower for keyword in info["keywords"]):
                    for keyword in info["keywords"]:
                        if keyword in ent_lower:
                            products.add(keyword)
    
    # Method 3: Noun phrases for complex products - be more selective
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for category, info in PRODUCT_CATEGORIES.items():
            for keyword in info["keywords"]:
                if keyword in chunk_text:
                    products.add(keyword)
                    break
    
    # Clean up products
    cleaned_products = []
    for product in products:
        if len(product.split()) <= 2:
            cleaned_products.append(product)
    
    result["products"] = list(set(cleaned_products))
    
    # 2. Extract features using regex patterns
    for feature_type, pattern in FEATURE_PATTERNS.items():
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    value = next((g for g in match if g), "")
                    if value:
                        result["features"][feature_type].append(value)
                else:
                    result["features"][feature_type].append(match)
    
    # 3. Extract price information using enhanced function
    price_info = extract_price_info(text)
    if price_info:
        for price in price_info:
            price_str = f"{price['range_type']} {price['amount']} {price['currency']}"
            result["features"]["price"].append(price_str)
    
    # 4. Extract intent
    intent_keywords = {
        "buy": ["buy", "purchase", "order", "get", "need", "want"],
        "compare": ["compare", "difference", "better", "vs", "versus"],
        "search": ["find", "look for", "search", "show me", "display", "list"],
        "recommend": ["recommend", "suggest", "advice", "best", "good"]
    }
    
    for intent, keywords in intent_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            result["intent"] = intent
            break
    
    # 5. Handle refinement queries
    if not result["products"] and previous_context:
        refinement_keywords = ["in", "with", "color", "size", "brand", "material", "style"]
        if any(keyword in text.lower() for keyword in refinement_keywords):
            result["products"] = previous_context.get("products", [])
            result["intent"] = "search"
    
    return result

def generate_human_response(combined_context, user_text, recommendations):
    """Generate natural, human-like conversational response"""
    
    # Determine response type
    products = combined_context.get('products', [])
    features = combined_context.get('features', {})
    
    # Check if this is a greeting or first message
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(word in user_text.lower() for word in greeting_words):
        return random.choice(RESPONSE_TEMPLATES["greeting"])
    
    # Build response based on context
    response_parts = []
    
    # Product-specific response
    if products:
        main_product = products[0]
        
        # Brand-specific response
        if features.get('brand'):
            brand = features['brand'][0]
            response_parts.append(
                random.choice(RESPONSE_TEMPLATES["brand_specific"])
                .format(brand=brand, product=main_product)
            )
        else:
            response_parts.append(
                random.choice(RESPONSE_TEMPLATES["product_found"])
                .format(product=main_product)
            )
    
    # Price-specific response
    if features.get('price'):
        price_info = features['price'][0]
        price_response = random.choice(RESPONSE_TEMPLATES["price_range"])
        
        # Format price range text
        if 'under' in price_info:
            price_range = f"under {price_info.split('under')[1].strip()}"
        elif 'over' in price_info:
            price_range = f"over {price_info.split('over')[1].strip()}"
        elif 'around' in price_info:
            price_range = f"around {price_info.split('around')[1].strip()}"
        else:
            price_range = f"in your price range"
        
        response_parts.append(price_response.format(price_range=price_range))
    
    # Color or other features
    if features.get('color'):
        color = features['color'][0]
        response_parts.append(f"I've focused on {color} options for you.")
    
    # If no specific response built, use generic
    if not response_parts:
        response_parts.append(random.choice(RESPONSE_TEMPLATES["multiple_options"]))
    
    # Add helpful ending
    helpful_endings = [
        "Here are my top recommendations:",
        "Check out these great options:",
        "Here's what I found for you:",
        "These should be perfect for you:",
        "Take a look at these options:"
    ]
    
    response_parts.append(random.choice(helpful_endings))
    
    return " ".join(response_parts)

def generate_smart_recommendations(combined_context, user_text):
    """Generate intelligent product recommendations"""
    
    # Get context information
    products = combined_context.get('products', [])
    features = combined_context.get('features', {})
    
    if not products:
        return ["Nike Air Max - Classic comfort", "Adidas Ultra Boost - Premium performance", "Puma RS-X - Modern style"]
    
    main_product = products[0]
    recommendations = []
    
    # Get available features
    brands = features.get('brand', [])
    colors = features.get('color', [])
    sizes = features.get('size', [])
    materials = features.get('material', [])
    
    # Popular brands for fallback
    popular_brands = {
        'shoes': ['Nike', 'Adidas', 'Puma'],
        'sneakers': ['Nike', 'Adidas', 'Puma'],
        'shirt': ['Zara', 'H&M', 'Uniqlo'],
        'jeans': ['Levi\'s', 'Wrangler', 'Lee'],
        'phone': ['Apple', 'Samsung', 'OnePlus'],
        'laptop': ['Dell', 'HP', 'Lenovo']
    }
    
    # Use specified brands or fallback to popular ones
    if brands:
        available_brands = brands
    else:
        available_brands = popular_brands.get(main_product, ['Nike', 'Adidas', 'Puma'])
    
    # Generate recommendations
    for i, brand in enumerate(available_brands[:3]):
        if colors and i == 0:
            recommendations.append(f"{brand} {main_product} - {colors[0]} color")
        elif sizes and i == 1:
            recommendations.append(f"{brand} {main_product} - Size {sizes[0]}")
        elif materials and i == 2:
            recommendations.append(f"{brand} {main_product} - {materials[0]} material")
        else:
            # Generic features
            generic_features = ["Premium quality", "Comfortable fit", "Durable design", "Stylish look", "Great value"]
            recommendations.append(f"{brand} {main_product} - {generic_features[i % len(generic_features)]}")
    
    return recommendations[:3]

def build_search_urls(combined_context):
    """Build optimized search URLs based on combined context"""
    query_parts = []
    
    # Add products
    if combined_context['products']:
        main_products = [p for p in combined_context['products'] if len(p.split()) <= 2][:2]
        query_parts.extend(main_products)
    
    # Add brand
    if 'brand' in combined_context['features']:
        query_parts.extend(combined_context['features']['brand'][:1])
    
    # Add color
    if 'color' in combined_context['features']:
        query_parts.extend(combined_context['features']['color'][:1])
    
    # Add size
    if 'size' in combined_context['features']:
        query_parts.extend(combined_context['features']['size'][:1])
    
    #Add price
    price_filters = combined_context['features'].get('price', [])
    price_query = ""
    for price in price_filters:
        if "under" in price:
            # Extract the numeric value (e.g., "under 1000 usd")
            amount = ''.join(filter(str.isdigit, price))
            price_query = f"under {amount}"
            break  # Use the first price filter found

    if price_query:
        query_parts.append(price_query)

    # Create query string
    unique_parts = []
    for part in query_parts:
        if part not in unique_parts:
            unique_parts.append(part)
    
    encoded_parts = [urllib.parse.quote_plus(str(part)) for part in query_parts]
    query = "+".join(encoded_parts) if encoded_parts else "products"
    
    return {
        "amazon": f"https://www.amazon.in/s?k={query}",
        "flipkart": f"https://www.flipkart.com/search?q={query}",
        "myntra": f"https://www.myntra.com/{query}",
        "ajio": f"https://www.ajio.com/search/?text={query}"
    }

# =============== Flask App ================
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Enhanced Shopping Assistant running. Use POST /chat."})

@app.route("/chat", methods=["POST"])
def chat():
    global session
    
    try:
        data = request.get_json()
        user_text = data.get("text", "")
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        # Extract product information from current message
        current_info = extract_product_info(user_text, session.current_search_context)
        
        # Update session context with new information
        session.update_context(current_info)
        
        # Use combined context for recommendations and URLs
        combined_context = session.current_search_context
        
        # Generate smart recommendations
        recommendations = generate_smart_recommendations(combined_context, user_text)
        
        # Build search URLs
        search_urls = build_search_urls(combined_context)
        
        # Generate human-like response
        conversational_reply = generate_human_response(combined_context, user_text, recommendations)
        
        # Update conversation history
        session.add_message(user_text, conversational_reply)
        
        # Return comprehensive response
        return jsonify({
            "reply": conversational_reply,
            "product_analysis": {
                "products": combined_context['products'],
                "features": dict(combined_context['features']),
                "context": combined_context['context'],
                "intent": combined_context['intent']
            },
            "recommendations": recommendations,
            "search_urls": search_urls,
            "conversation_history": session.conversation_history
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 400
    
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Check if required libraries are available
        try:
            import librosa
            import numpy as np
            from tempfile import NamedTemporaryFile
            import os
        except ImportError as e:
            return jsonify({
                "error": f"Missing required library: {str(e)}",
                "suggestions": [
                    "Install librosa: pip install librosa",
                    "Install numpy: pip install numpy",
                    "Or try: pip install librosa numpy soundfile"
                ]
            }), 500
        
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            print(f"Processing audio file: {tmp_path}")
            
            # Load audio with librosa
            try:
                audio, sr = librosa.load(tmp_path, sr=16000)  # Whisper expects 16kHz
                print(f"Audio loaded successfully - Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}")
            except Exception as e:
                return jsonify({
                    "error": f"Failed to load audio file: {str(e)}",
                    "suggestions": [
                        "Try a different audio format (WAV, MP3, M4A)",
                        "Check if the audio file is not corrupted",
                        "Ensure the file is not too large (< 25MB recommended)"
                    ]
                }), 400
            
            # Check audio duration
            duration = len(audio) / sr
            if duration < 0.1:
                return jsonify({
                    "error": "Audio is too short (< 0.1 seconds)",
                    "suggestions": [
                        "Record for at least 1-2 seconds",
                        "Check if the audio file contains actual sound"
                    ]
                }), 400
            
            if duration > 30:  # 30 second limit for this demo
                return jsonify({
                    "error": "Audio is too long (> 30 seconds)",
                    "suggestions": [
                        "Split the audio into shorter segments",
                        "Use a shorter recording"
                    ]
                }), 400
            
            # Convert to the format expected by the pipeline
            try:
                print("Starting transcription...")
                result = asr_pipeline(audio)
                transcription = result["text"]
                print(f"Transcription completed: {transcription}")
                
                # Return successful response
                return jsonify({
                    "transcription": transcription,
                    "audio_length": duration,
                    "confidence": "N/A",  # Whisper doesn't provide confidence scores directly
                    "success": True
                })
                
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                return jsonify({
                    "error": f"Transcription failed: {str(e)}",
                    "suggestions": [
                        "Try speaking more clearly",
                        "Ensure good audio quality",
                        "Check if the audio contains speech",
                        "Try recording in a quieter environment"
                    ]
                }), 500
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
                print(f"Temporary file cleaned up: {tmp_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temp file: {e}")
                
    except Exception as e:
        print(f"Unexpected error in transcription: {str(e)}")
        return jsonify({
            "error": f"Unexpected error: {str(e)}",
            "suggestions": [
                "Check server logs for more details",
                "Try restarting the Flask application",
                "Ensure all dependencies are installed"
            ]
        }), 500

# Add a test endpoint to check if everything is working
@app.route("/transcribe/test", methods=["GET"])
def test_transcription():
    """Test endpoint to verify transcription setup"""
    try:
        import librosa
        import numpy as np
        
        # Test if the ASR pipeline is working
        test_audio = np.random.randn(16000)  # 1 second of random noise
        result = asr_pipeline(test_audio)
        
        return jsonify({
            "status": "healthy",
            "librosa_version": librosa.__version__,
            "numpy_version": np.__version__,
            "asr_pipeline_working": True,
            "test_transcription": result["text"]
        })
        
    except ImportError as e:
        return jsonify({
            "status": "error",
            "error": f"Missing library: {str(e)}",
            "suggestions": [
                "Install missing dependencies",
                "Run: pip install librosa numpy soundfile"
            ]
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"ASR pipeline error: {str(e)}",
            "suggestions": [
                "Check Whisper model installation",
                "Try restarting the application"
            ]
        }), 500

@app.route("/reset", methods=["POST"])
def reset_session():
    global session
    session.reset()
    return jsonify({"message": "Session reset successfully"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)