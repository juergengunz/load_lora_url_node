import os
import hashlib
import requests
import folder_paths
import comfy.utils
import comfy.sd
from tqdm import tqdm

class LoadLoraFromURL:
    """Load a LoRA model from a URL"""
    
    MAX_CACHE_SIZE = 20
    
    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_loras")
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def _enforce_cache_limit(self):
        """Remove oldest cached files if cache exceeds MAX_CACHE_SIZE"""
        cached_files = [
            os.path.join(self.cache_dir, f) 
            for f in os.listdir(self.cache_dir) 
            if f.endswith(".safetensors")
        ]
        
        if len(cached_files) <= self.MAX_CACHE_SIZE:
            return
        
        # Sort by modification time (oldest first)
        cached_files.sort(key=lambda f: os.path.getmtime(f))
        
        # Remove oldest files until we're at the limit
        files_to_remove = len(cached_files) - self.MAX_CACHE_SIZE
        for f in cached_files[:files_to_remove]:
            print(f"Removing oldest cached LoRA: {os.path.basename(f)}")
            os.remove(f)

    def download_if_needed(self, url):
        """Download the file if not in cache"""
        filename = hashlib.md5(url.encode()).hexdigest() + ".safetensors"
        local_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(local_path):
            return local_path
        
        # Enforce cache limit before downloading new file
        self._enforce_cache_limit()
            
        print(f"Downloading LoRA from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                pbar.update(size)
                
        return local_path

    def load_lora(self, url, model, strength):
        try:
            # Download or get cached file
            lora_path = self.download_if_needed(url)
            
            # Load the LoRA using ComfyUI's built-in functions
            lora = comfy.utils.load_torch_file(lora_path)
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength, 0
            )
            return (model_lora,)
            
        except Exception as e:
            print(f"Error loading LoRA: {str(e)}")
            return (model,)