import os

# Memory optimizations for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
import tempfile
from contextlib import asynccontextmanager
import re

# Detect device (Mac with M1/M2/M3/M4)
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
map_location = torch.device(device)

# Patch torch.load for MPS compatibility if needed
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading ChatterboxTTS model on {device}...")
    # Passing attn_implementation="eager" to avoid Transformers warnings/errors on some texts
    try:
        model = ChatterboxTTS.from_pretrained(device=device, attn_implementation="eager")
    except TypeError:
        # Fallback if the specific loader doesn't accept that kwarg directly
        model = ChatterboxTTS.from_pretrained(device=device)
    yield
    # Cleanup if necessary

app = FastAPI(title="Chatterbox TTS API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device, "model_loaded": model is not None}

def chunk_text(text, max_chars=300):
    """Chunks text by periods directly."""
    # Split by periods followed by optional whitespace
    sentences = re.split(r'\.\s*', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add the period back for the model
        sentence = sentence + "."
            
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single sentence is still too long, just hard split it
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

@app.post("/generate")
async def generate_audio(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Use local preview.mp3 as the prompt path
    prompt_path = "preview.mp3"
    if not os.path.exists(prompt_path):
        raise HTTPException(status_code=500, detail="Default audio prompt (preview.mp3) not found on server")

    try:
        # Re-enabled chunking with 300 character limit
        chunks = chunk_text(text, max_chars=300)
        print(f"Generating audio for {len(chunks)} chunks...")
        
        wav_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            wav = model.generate(
                chunk,
                audio_prompt_path=prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            wav_chunks.append(wav)
        
        if not wav_chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from text")
            
        # Concatenate audio chunks
        final_wav = torch.cat(wav_chunks, dim=-1)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_out:
            output_path = tmp_out.name
        
        ta.save(output_path, final_wav, model.sr)
        
        def iterfile():
            try:
                with open(output_path, "rb") as f:
                    yield from f
            finally:
                # Cleanup output file after streaming
                if os.path.exists(output_path):
                    os.remove(output_path)

        return StreamingResponse(iterfile(), media_type="audio/mpeg", headers={
            "Content-Disposition": "attachment; filename=generated.mp3"
        })

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # landing page
    return """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatterbox x FastAPI</title>

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet">
        <style>
            ul {
                width: 50vw;
                margin: 0;
                padding: 0;
            }

            li {
                list-style: none;
                background-color: #eee;
                padding: 10px;
                margin-bottom: 3px;
                line-height: 150%;
            }

            li:hover {
                background-color: #ccc;
            }

            li code {
                font-size: smaller;
                color: white;
                background-color: green;
                padding: 3px 5px;
                border-radius: 5px;
                vertical-align: middle;
            }

            li i {
                font-size: smaller;
                font-style: italic;
            }

            li:hover a {
                font-weight: 500;
                text-decoration: underline;
            }

            a {
                color: blue;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }

            .inter-medium {
                font-family: "Inter", sans-serif;
                font-optical-sizing: auto;
                font-weight: 500;
                font-style: normal;
            }

            .inter-light {
                font-family: "Inter", sans-serif;
                font-optical-sizing: auto;
                font-weight: 300;
                font-style: normal;
            }
        </style>
    </head>
    <body>
        <div>
            <h1 class="inter-medium">A quick API experiment to test returning audio data over http</h1>
            <p class="inter-light">Purpose: API backend for Chatterbox TTS model</p>
            <ul>
                <li class="inter-light"><a href="/docs">API docs</a></li>
                <li class="inter-light"><a href="/generate">Generate MP3 audio</a> <code>POST</code><br />
                <i class="inter-light">Currently requires an audio file called <span class="inter-medium">preview.mp3</span> to be in the root of the application.</i></li>
                <li class="inter-light"><a href="/health">Check health</a></li>
            </ul>
        </div> 
        <script type="text/javascript">
            const listItems = document.querySelectorAll("li");

            listItems.forEach(li => {
                li.style.cursor = "pointer"; // Make it look clickable
                
                li.addEventListener("click", () => {
                    const anchor = li.querySelector("a");
                    const postBlock = li.querySelector("code");

                    if (anchor && anchor.href && !postBlock) {
                        window.location.href = anchor.href;
                    }
                });
            });
        </script>
    </body>
</html>
    """
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
