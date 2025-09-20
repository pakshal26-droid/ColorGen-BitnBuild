import os
import json
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
import colorsys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import uvicorn

# Load environment
load_dotenv()

class ColorInfo(BaseModel):
    hex: str
    rgb: List[int]
    hsl: List[int]
    name: str = ""

class PaletteResponse(BaseModel):
    primary: List[ColorInfo]
    secondary: List[ColorInfo]
    accent: List[ColorInfo]
    accessibility_score: float
    wcag_compliance: Dict[str, bool]
    colorblind_safe: bool
    method_used: str

class ColorUtils:
    @staticmethod
    def rgb_to_hex(rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    @staticmethod
    def rgb_to_hsl(rgb):
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return [int(h*360), int(s*100), int(l*100)]
    
    @staticmethod
    def create_color_info(rgb, name=""):
        return ColorInfo(
            hex=ColorUtils.rgb_to_hex(rgb),
            rgb=list(rgb),
            hsl=ColorUtils.rgb_to_hsl(rgb),
            name=name
        )
    
    @staticmethod
    def contrast_ratio(color1, color2):
        def luminance(rgb):
            r, g, b = [x/255.0 for x in rgb]
            r = r/12.92 if r <= 0.03928 else pow((r+0.055)/1.055, 2.4)
            g = g/12.92 if g <= 0.03928 else pow((g+0.055)/1.055, 2.4)
            b = b/12.92 if b <= 0.03928 else pow((b+0.055)/1.055, 2.4)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1, l2 = luminance(color1), luminance(color2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

# =============================================================================
# GAN/VAE MODEL (Load your pre-trained Keras models)
# =============================================================================

class ColorVAE:
    def __init__(self, models_path="weights"):
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.models_loaded = False
        self.models_path = models_path
        
        # Try to load the models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained Keras models"""
        try:
            # Load encoder
            encoder_path = os.path.join(self.models_path, "encoder.h5")
            if os.path.exists(encoder_path):
                self.encoder = keras.models.load_model(encoder_path, compile=False)
                print("✅ Encoder loaded")
            
            # Load decoder (try both decoder.h5 and decoder2.h5)
            decoder_path = os.path.join(self.models_path, "decoder.h5")
            decoder2_path = os.path.join(self.models_path, "decoder2.h5")
            
            if os.path.exists(decoder_path):
                self.decoder = keras.models.load_model(decoder_path, compile=False)
                print("✅ Decoder loaded")
            elif os.path.exists(decoder2_path):
                self.decoder = keras.models.load_model(decoder2_path, compile=False)
                print("✅ Decoder2 loaded")
            
            # Load discriminator (optional for generation)
            discriminator_path = os.path.join(self.models_path, "discriminator.h5")
            if os.path.exists(discriminator_path):
                self.discriminator = keras.models.load_model(discriminator_path, compile=False)
                print("✅ Discriminator loaded")
            
            # Check if we have minimum required models
            if self.encoder and self.decoder:
                self.models_loaded = True
                print("🎨 Color VAE models ready!")
            else:
                print("❌ Missing required models (encoder/decoder)")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def generate_palette(self, seed_colors):
        """Generate refined palette from seed colors using VAE"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
        
        try:
            # Prepare input - normalize colors to [0,1]
            input_colors = seed_colors.astype(np.float32) / 255.0
            
            # Reshape for model input (adjust based on your model's expected input shape)
            # Common shapes: (batch, colors*3) or (batch, colors, 3)
            if len(input_colors.shape) == 2:  # (colors, 3)
                # Try flattened input first
                model_input = input_colors.flatten().reshape(1, -1)
            else:
                model_input = input_colors.reshape(1, -1)
            
            # Encode to latent space
            if hasattr(self.encoder, 'predict'):
                latent = self.encoder.predict(model_input, verbose=0)
            else:
                latent = self.encoder(model_input)
            
            # Handle VAE latent space (mean, log_var) if present
            if isinstance(latent, list) and len(latent) == 2:
                # VAE case: use mean
                latent = latent[0]
            
            # Decode back to color space
            if hasattr(self.decoder, 'predict'):
                generated = self.decoder.predict(latent, verbose=0)
            else:
                generated = self.decoder(latent)
            
            # Post-process output
            generated = np.array(generated).squeeze()
            
            # Ensure we have the right shape and scale
            if generated.max() <= 1.0:  # If normalized
                generated = generated * 255.0
            
            generated = np.clip(generated, 0, 255).astype(np.uint8)
            
            # Reshape to (n_colors, 3) if needed
            if len(generated.shape) == 1:
                generated = generated.reshape(-1, 3)
            
            return generated
            
        except Exception as e:
            print(f"VAE generation failed: {e}")
            # Fallback to returning input colors
            return seed_colors

# =============================================================================
# IMAGE COLOR EXTRACTOR
# =============================================================================

class ImageColorExtractor:
    def __init__(self, models_path="weights"):
        self.vae = None
        self.models_loaded = False
        
        # Try to load VAE/GAN models
        if models_path and os.path.exists(models_path):
            try:
                self.vae = ColorVAE(models_path)
                self.models_loaded = self.vae.models_loaded
                if self.models_loaded:
                    print("✅ VAE models loaded successfully!")
                else:
                    print("📝 Will use K-means fallback")
            except Exception as e:
                print(f"❌ Failed to load VAE: {e}")
                print("📝 Will use K-means fallback")
        else:
            print("📝 No models folder found, using K-means fallback")
    
    def extract_colors_kmeans(self, image_path, n_colors=7):
        """Extract colors using K-means clustering"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        h, w = img.shape[:2]
        if w > 400:
            scale = 400/w
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # Reshape for clustering
        pixels = img.reshape(-1, 3)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        
        # Sort by cluster size
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]
        
        return colors[sorted_indices].astype(int)
    
    def extract_colors_vae(self, image_path):
        """Extract colors using VAE refinement"""
        # First get base colors with K-means
        base_colors = self.extract_colors_kmeans(image_path, n_colors=5)
        
        # Refine with VAE
        refined_colors = self.vae.generate_palette(base_colors)
        return refined_colors
    
    def extract_palette(self, image_path):
        """Main extraction method"""
        if self.models_loaded:
            try:
                colors = self.extract_colors_vae(image_path)
                method = "VAE"
            except Exception as e:
                print(f"VAE failed: {e}, using K-means")
                colors = self.extract_colors_kmeans(image_path)
                method = "K-means"
        else:
            colors = self.extract_colors_kmeans(image_path)
            method = "K-means"
        
        # Convert to ColorInfo objects
        palette = []
        for i, color in enumerate(colors):
            palette.append(ColorUtils.create_color_info(color, f"Color_{i+1}"))
        
        return palette, method

# =============================================================================
# LLM PALETTE GENERATOR
# =============================================================================

class LLMGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def generate_from_prompt(self, prompt, base_colors=None):
        """Generate palette from text prompt"""
        
        system_msg = """You are a color palette expert. Generate a JSON response with this structure:
{
    "primary": [{"rgb": [r,g,b], "name": "color_name"}, ...],
    "secondary": [{"rgb": [r,g,b], "name": "color_name"}, ...], 
    "accent": [{"rgb": [r,g,b], "name": "color_name"}, ...]
}

Rules:
- Primary: 2-3 main colors
- Secondary: 2-3 supporting colors  
- Accent: 1-2 highlight colors
- RGB values 0-255
- Descriptive color names
- Ensure harmony and good contrast"""

        user_msg = f"Create a palette for: {prompt}"
        
        if base_colors:
            base_desc = ", ".join([f"{c.name} ({c.hex})" for c in base_colors])
            user_msg += f"\n\nRefine these image colors: {base_desc}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            palette_data = json.loads(content)
            
            # Convert to ColorInfo objects
            result = {}
            for category in ['primary', 'secondary', 'accent']:
                if category in palette_data:
                    colors = []
                    for color_data in palette_data[category]:
                        rgb = color_data['rgb']
                        name = color_data.get('name', f'{category}_color')
                        colors.append(ColorUtils.create_color_info(rgb, name))
                    result[category] = colors
            
            return result
            
        except Exception as e:
            print(f"LLM failed: {e}")
            return self._fallback_palette()
    
    def _fallback_palette(self):
        """Simple fallback when LLM fails"""
        return {
            'primary': [ColorUtils.create_color_info([52, 152, 219], "blue"), 
                       ColorUtils.create_color_info([231, 76, 60], "red")],
            'secondary': [ColorUtils.create_color_info([149, 165, 166], "gray"),
                         ColorUtils.create_color_info([39, 174, 96], "green")],
            'accent': [ColorUtils.create_color_info([241, 196, 15], "yellow")]
        }

# =============================================================================
# ACCESSIBILITY AUDITOR
# =============================================================================

class AccessibilityAuditor:
    @staticmethod
    def audit_palette(palette_dict):
        """Audit palette for accessibility"""
        all_colors = []
        for category in palette_dict.values():
            all_colors.extend([c.rgb for c in category])
        
        # Check all color pairs
        total_pairs = 0
        aa_compliant = 0
        aaa_compliant = 0
        
        for i, color1 in enumerate(all_colors):
            for color2 in all_colors[i+1:]:
                ratio = ColorUtils.contrast_ratio(color1, color2)
                total_pairs += 1
                if ratio >= 4.5:
                    aa_compliant += 1
                if ratio >= 7.0:
                    aaa_compliant += 1
        
        if total_pairs == 0:
            return {'score': 0, 'aa': False, 'aaa': False}
        
        aa_rate = aa_compliant / total_pairs
        aaa_rate = aaa_compliant / total_pairs
        score = aa_rate * 0.7 + aaa_rate * 0.3
        
        return {
            'score': score,
            'aa': aa_rate > 0.5,
            'aaa': aaa_rate > 0.3
        }
    
    @staticmethod
    def simulate_colorblind(rgb, cb_type):
        """Simple colorblind simulation"""
        r, g, b = rgb
        
        if cb_type == 'protanopia':  # Red-blind
            return [int(0.567*r + 0.433*g), int(0.558*r + 0.442*g), int(0.242*g + 0.758*b)]
        elif cb_type == 'deuteranopia':  # Green-blind  
            return [int(0.625*r + 0.375*g), int(0.700*r + 0.300*g), int(0.300*g + 0.700*b)]
        elif cb_type == 'tritanopia':  # Blue-blind
            return [int(0.950*r + 0.050*g), int(0.433*g + 0.567*b), int(0.475*g + 0.525*b)]
        return rgb

# =============================================================================
# MAIN CHROMAGEN PIPELINE
# =============================================================================

class ChromaGenPipeline:
    def __init__(self, openai_key, models_path="weights"):
        self.extractor = ImageColorExtractor(models_path)
        self.llm = LLMGenerator(openai_key)
        self.auditor = AccessibilityAuditor()
    
    def generate_palette(self, text_prompt=None, image_path=None, hybrid=True):
        """Main palette generation method"""
        base_colors = None
        method_used = []
        
        # Extract from image
        if image_path:
            base_colors, extraction_method = self.extractor.extract_palette(image_path)
            method_used.append(f"Image({extraction_method})")
        
        # Generate/refine with LLM  
        if text_prompt:
            if hybrid and base_colors:
                palette_dict = self.llm.generate_from_prompt(text_prompt, base_colors)
                method_used.append("LLM-Hybrid")
            else:
                palette_dict = self.llm.generate_from_prompt(text_prompt)
                method_used.append("LLM-Only")
        elif base_colors:
            # Organize extracted colors
            palette_dict = self._organize_colors(base_colors)
            method_used.append("Image-Only")
        else:
            raise ValueError("Need either text prompt or image")
        
        # Audit accessibility
        audit_results = self.auditor.audit_palette(palette_dict)
        
        return PaletteResponse(
            primary=palette_dict.get('primary', []),
            secondary=palette_dict.get('secondary', []),
            accent=palette_dict.get('accent', []),
            accessibility_score=audit_results['score'],
            wcag_compliance={'aa': audit_results['aa'], 'aaa': audit_results['aaa']},
            colorblind_safe=audit_results['score'] > 0.6,
            method_used=" + ".join(method_used)
        )
    
    def _organize_colors(self, colors):
        """Organize extracted colors into categories"""
        return {
            'primary': colors[:3],
            'secondary': colors[3:5] if len(colors) > 3 else [],
            'accent': colors[5:7] if len(colors) > 5 else []
        }

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="ChromaGen", description="AI Color Palette Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline - moved to module level
openai_key = os.getenv('OPENAI_API_KEY')
pipeline = None
if openai_key:
    pipeline = ChromaGenPipeline(openai_key, models_path="weights")

@app.get("/")
def root():
    return {
        "message": "ChromaGen AI Color Palette Generator", 
        "status": "ready" if pipeline else "not ready",
        "endpoints": ["/health", "/generate/text", "/generate/image", "/generate/hybrid"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "has_openai_key": bool(os.getenv('OPENAI_API_KEY')),
        "models_loaded": pipeline.extractor.models_loaded if pipeline else False
    }

@app.post("/generate/text", response_model=PaletteResponse)
def generate_text(prompt: str):
    """Generate from text prompt only"""
    if not pipeline:
        raise HTTPException(500, "Pipeline not ready - check OpenAI API key")
    
    try:
        print(f"🔍 Generating palette for: '{prompt}'")
        result = pipeline.generate_palette(text_prompt=prompt)
        print(f"✅ Generated successfully!")
        return result
    except Exception as e:
        print(f"❌ Error in generate_text: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.post("/generate/image", response_model=PaletteResponse)  
def generate_image(file: UploadFile = File(...)):
    """Generate from image only"""
    if not pipeline:
        raise HTTPException(500, "Pipeline not ready")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Must be an image file")
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        result = pipeline.generate_palette(image_path=tmp_path)
        os.unlink(tmp_path)  # cleanup
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

@app.post("/generate/hybrid", response_model=PaletteResponse)
def generate_hybrid(prompt: str = Form(...), file: UploadFile = File(...)):
    """Generate using both text + image"""
    if not pipeline:
        raise HTTPException(500, "Pipeline not ready")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Must be an image file")
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        result = pipeline.generate_palette(text_prompt=prompt, image_path=tmp_path, hybrid=True)
        os.unlink(tmp_path)  # cleanup
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)