import os
import hashlib
import time
import requests
import folder_paths
import comfy.utils
import comfy.sd
from tqdm import tqdm

class LoadLoraFromURL:
    """Load a LoRA model from a URL"""
    
    MAX_CACHE_SIZE = 20
    MAX_DOWNLOAD_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 2
    REQUEST_TIMEOUT_SECONDS = 60
    
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
            },
            "optional": {
                "on_failure": (["abort_workflow", "continue_without_lora"],),
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
        temp_path = local_path + ".part"
        
        if os.path.exists(local_path):
            return local_path
        
        # Enforce cache limit before downloading new file
        self._enforce_cache_limit()
            
        for attempt in range(1, self.MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                print(
                    f"Downloading LoRA from {url} "
                    f"(attempt {attempt}/{self.MAX_DOWNLOAD_ATTEMPTS})"
                )
                with requests.get(
                    url,
                    stream=True,
                    timeout=self.REQUEST_TIMEOUT_SECONDS
                ) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))

                    with open(temp_path, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True
                    ) as pbar:
                        for data in response.iter_content(chunk_size=8192):
                            if not data:
                                continue
                            size = f.write(data)
                            pbar.update(size)

                os.replace(temp_path, local_path)
                return local_path

            except requests.RequestException as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if attempt == self.MAX_DOWNLOAD_ATTEMPTS:
                    raise

                print(
                    f"Download failed: {str(e)}. "
                    f"Retrying in {self.RETRY_DELAY_SECONDS} seconds..."
                )
                time.sleep(self.RETRY_DELAY_SECONDS)

        return local_path

    def load_lora(self, url, model, strength, on_failure="abort_workflow"):
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
            message = f"Error loading LoRA from URL: {str(e)}"
            if on_failure == "abort_workflow":
                raise RuntimeError(message) from e

            print(message)
            return (model,)
