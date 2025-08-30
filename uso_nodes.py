# uso_nodes.py (Version finale avec stratégie FP8)

import torch
import numpy as np
import os
from PIL import Image
from dataclasses import dataclass
import comfy.utils
import comfy.model_management as model_management
import folder_paths

# Imports pour le téléchargement et le chargement des modèles
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoConfig, T5EncoderModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from safetensors.torch import load_file as load_sft_file, save_file

# Imports depuis votre structure de package "uso"
from .uso.flux.util import configs, load_sft as load_sft_from_uso, set_lora
from .uso.flux.model import Flux
from .uso.flux.modules.autoencoder import AutoEncoder
from .uso.flux.modules.conditioner import HFEmbedder
from .uso.flux.sampling import get_noise, get_schedule, prepare_multi_ip, unpack
from .uso.flux.pipeline import preprocess_ref
from transformers import SiglipVisionModel, SiglipImageProcessor


# MODIFICATION: T5Embedder doit maintenant s'assurer que la sortie est en bfloat16
# pour le reste de la pipeline, même si le modèle T5 est en FP8.
class T5Embedder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device

    def __call__(self, text, max_length=256):
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True,
            return_tensors="pt", max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # On s'assure que la sortie est bien en bfloat16, car le reste du code l'attend.
            last_hidden_state = outputs.last_hidden_state.to(dtype=torch.bfloat16)
        return last_hidden_state

@dataclass
class USOModel:
    flux_model: object
    ae_model: object
    t5_embedder: T5Embedder
    clip_model: object
    siglip_model: object
    siglip_processor: object


class USOLoader:
    _models = {}
    USO_MODELS_PATH = os.path.join(folder_paths.models_dir, "USO_models")

    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(cls.USO_MODELS_PATH, exist_ok=True)
        try:
            files_list = sorted([f for f in os.listdir(cls.USO_MODELS_PATH) if os.path.isfile(os.path.join(cls.USO_MODELS_PATH, f)) and f.endswith('.safetensors')])
        except Exception: files_list = []
        if not files_list: files_list.append("Placez vos modèles .safetensors ici !")
        return { "required": {
                "flux_model_name": (files_list, ), "ae_model_name": (files_list, ),
                "uso_lora_name": (files_list, ), "uso_projector_name": (files_list, ),
            }}

    RETURN_TYPES = ("USO_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "USO"

    def download_models(self):
        # ... (code de téléchargement inchangé)
        t5_path = os.path.join(self.USO_MODELS_PATH, "t5xxl_fp8_e4m3fn_scaled.safetensors")
        if not os.path.exists(t5_path):
            print(f"[USO Node] Téléchargement T5...")
            hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp8_e4m3fn_scaled.safetensors", local_dir=self.USO_MODELS_PATH, local_dir_use_symlinks=False)
        clip_path = os.path.join(self.USO_MODELS_PATH, "clip-vit-l14")
        if not os.path.exists(clip_path):
            print(f"[USO Node] Téléchargement CLIP...")
            snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir=clip_path, local_dir_use_symlinks=False)
        siglip_path = os.path.join(self.USO_MODELS_PATH, "siglip-so400m-patch14-384")
        if not os.path.exists(siglip_path):
            print(f"[USO Node] Téléchargement SigLIP...")
            snapshot_download(repo_id="google/siglip-so400m-patch14-384", local_dir=siglip_path, local_dir_use_symlinks=False)
        return t5_path, clip_path, siglip_path

    def load_model(self, flux_model_name, ae_model_name, uso_lora_name, uso_projector_name):
        cache_key = (flux_model_name, ae_model_name, uso_lora_name, uso_projector_name)
        if cache_key in self._models:
            return (self._models[cache_key],)

        t5_path, clip_path, siglip_path = self.download_models()
        flux_path = os.path.join(self.USO_MODELS_PATH, flux_model_name)
        ae_path = os.path.join(self.USO_MODELS_PATH, ae_model_name)
        uso_lora_path = os.path.join(self.USO_MODELS_PATH, uso_lora_name)
        uso_projector_path = os.path.join(self.USO_MODELS_PATH, uso_projector_name)

        device = model_management.get_torch_device()
        compute_dtype = torch.bfloat16
        
        # --- Chargement Flux & VAE en bfloat16 pour la compatibilité des calculs ---
        print("USO: Chargement FLUX & VAE...")
        model_spec = configs["flux-dev-fp8"]
        flux_model = Flux(model_spec.params)
        flux_model = set_lora(flux_model, lora_rank=128)
        base_sd = load_sft_from_uso(flux_path, device="cpu"); lora_sd = load_sft_from_uso(uso_lora_path, device="cpu"); proj_sd = load_sft_from_uso(uso_projector_path, device="cpu")
        lora_sd.update(proj_sd); base_sd.update(lora_sd)
        flux_model.load_state_dict(base_sd, strict=False)
        flux_model.to(device, dtype=compute_dtype).eval()
        del base_sd, lora_sd, proj_sd
        ae_model = AutoEncoder(configs["flux-dev-fp8"].ae_params)
        ae_sd = load_sft_from_uso(ae_path, device="cpu")
        ae_model.load_state_dict(ae_sd)
        ae_model.to(device, dtype=compute_dtype).eval()
        del ae_sd

        # --- Chargement T5 avec stratégie FP8 ---
        print("USO: Chargement du modèle T5 Encoder...")
        model_hub_name = "google/t5-v1_1-xxl"
        tokenizer = AutoTokenizer.from_pretrained(model_hub_name)
        config = AutoConfig.from_pretrained(model_hub_name)

        with init_empty_weights():
            t5_model = T5EncoderModel(config=config)
        t5_model.tie_weights()

        print("USO: Nettoyage des poids T5...")
        full_state_dict = load_sft_file(t5_path, device="cpu")
        cleaned_state_dict = {k: v for k, v in full_state_dict.items() if "scale_weight" not in k and "scaled_fp8" not in k}
        del full_state_dict

        # MODIFICATION: On recrée le fichier temporaire pour le passer à accelerate
        temp_cleaned_checkpoint = os.path.join(self.USO_MODELS_PATH, "temp_cleaned_t5.safetensors")
        save_file(cleaned_state_dict, temp_cleaned_checkpoint)
        del cleaned_state_dict

        offload_folder = os.path.join(self.USO_MODELS_PATH, "offload_cache")
        os.makedirs(offload_folder, exist_ok=True)
        
        # MODIFICATION: On essaie de charger en FP8 pour économiser la RAM/VRAM
        try:
            print("USO: Tentative de chargement de T5 en FP8 pour économiser la mémoire...")
            t5_dtype = torch.float8_e4m3fn
            t5_model = load_checkpoint_and_dispatch(
                t5_model, checkpoint=temp_cleaned_checkpoint, device_map="auto",
                no_split_module_classes=["T5Block"], dtype=t5_dtype, offload_folder=offload_folder
            )
        except Exception as e:
            print(f"USO: Echec du chargement en FP8 ({e}). Retour à bfloat16 (plus gourmand en mémoire)...")
            t5_dtype = torch.bfloat16
            t5_model = load_checkpoint_and_dispatch(
                t5_model, checkpoint=temp_cleaned_checkpoint, device_map="auto",
                no_split_module_classes=["T5Block"], dtype=t5_dtype, offload_folder=offload_folder
            )

        os.remove(temp_cleaned_checkpoint) # On supprime le fichier temporaire
        t5_model.eval()
        t5_embedder = T5Embedder(t5_model, tokenizer)
        print("USO: T5 Encoder chargé avec succès.")

        # --- Chargement des autres modèles ---
        print("USO: Chargement de CLIP & SigLIP...")
        clip_model = HFEmbedder(version=clip_path, max_length=77, torch_dtype=torch.bfloat16).to(device).eval()
        siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(device, dtype=compute_dtype).eval()
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        flux_model.vision_encoder = siglip_model

        print("USO: Tous les modèles ont été chargés avec succès.")
        uso_model_package = USOModel(
            flux_model=flux_model, ae_model=ae_model, t5_embedder=t5_embedder,
            clip_model=clip_model, siglip_model=siglip_model, siglip_processor=siglip_processor
        )
        self._models[cache_key] = uso_model_package
        return (uso_model_package,)


class USOSampler:
    @classmethod
    def INPUT_TYPES(cls):
        # ... (inchangé)
        return { "required": {
                "uso_model": ("USO_MODEL",), "prompt": ("STRING", {"multiline": True, "default": "A beautiful woman."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}), "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}), "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}), "content_ref_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 16}),
            }, "optional": { "content_image": ("IMAGE",), "style_image_1": ("IMAGE",), "style_image_2": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "sample"
    CATEGORY = "USO"

    def tensor_to_pil(self, tensor):
        # ... (inchangé)
        return Image.fromarray((tensor.squeeze(0).cpu().numpy()*255).astype(np.uint8))
    def pil_to_tensor(self, pil):
        # ... (inchangé)
        return torch.from_numpy(np.array(pil).astype(np.float32)/255.0).unsqueeze(0).permute(0,3,1,2)*2.0-1.0

    def sample(self, uso_model, prompt, seed, steps, width, height, guidance, content_ref_size,
               content_image=None, style_image_1=None, style_image_2=None):
        device = model_management.get_torch_device()
        t5_embedder = uso_model.t5_embedder
        
        ref_content_imgs = []
        if content_image is not None:
            ref_content_imgs.append(self.tensor_to_pil(content_image))

        siglip_inputs = []
        style_images_pil = []
        if style_image_1 is not None: style_images_pil.append(self.tensor_to_pil(style_image_1))
        if style_image_2 is not None: style_images_pil.append(self.tensor_to_pil(style_image_2))
        if style_images_pil:
            with torch.no_grad():
                siglip_inputs = [uso_model.siglip_processor(img, return_tensors="pt").to(device) for img in style_images_pil]
        
        x_1_refs = []
        if ref_content_imgs:
            with torch.no_grad():
                for ref_img in ref_content_imgs:
                    ref_pil = preprocess_ref(ref_img, content_ref_size)
                    ref_tensor = self.pil_to_tensor(ref_pil).to(device, dtype=torch.bfloat16)
                    
                    print("USO Sampler: Encodage de l'image de référence avec le VAE...")
                    encoded = uso_model.ae_model.encode(ref_tensor).to(torch.bfloat16)
                    x_1_refs.append(encoded)

            # --- MODIFICATION: OFFLOADING DU VAE ---
            # Le VAE a fini son travail pour l'instant, on le déplace sur le CPU pour libérer la VRAM
            print("USO Sampler: Libération de la VRAM en déplaçant le VAE sur le CPU.")
            uso_model.ae_model.to("cpu")
            comfy.model_management.soft_empty_cache() # On demande à PyTorch de nettoyer la VRAM

        pbar_comfy = comfy.utils.ProgressBar(steps)
        x_start = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)
        timesteps = get_schedule(steps, (width//8)*(height//8)//(16*16), shift=True)
        
        inp_cond = prepare_multi_ip(
            t5=t5_embedder, clip=uso_model.clip_model, img=x_start,
            prompt=prompt, ref_imgs=x_1_refs, pe="d",
        )
        
        img = inp_cond['img']
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

        print("USO Sampler: Démarrage de la boucle de diffusion principale...")
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            with torch.no_grad():
                pred = uso_model.flux_model(
                    img=img, img_ids=inp_cond['img_ids'],
                    ref_img=inp_cond.get('ref_img'), ref_img_ids=inp_cond.get('ref_img_ids'),
                    txt=inp_cond['txt'], txt_ids=inp_cond['txt_ids'],
                    y=inp_cond['vec'], timesteps=t_vec, guidance=guidance_vec,
                    siglip_inputs=siglip_inputs,
                )
            img = img + (t_prev - t_curr) * pred
            pbar_comfy.update(1)

        final_latent_unpacked = unpack(img.to(torch.float32), height, width)
        
        with torch.no_grad():
            # --- MODIFICATION: ON RAMÈNE LE VAE ---
            # On a de nouveau besoin du VAE, on le remet sur le GPU
            print("USO Sampler: Rapatriement du VAE sur le GPU pour le décodage final.")
            uso_model.ae_model.to(device)
            decoded_img = uso_model.ae_model.decode(final_latent_unpacked)
        
        output_image = (decoded_img.clamp(-1, 1) + 1.0) / 2.0
        return (output_image.permute(0, 2, 3, 1).cpu(), {"samples": final_latent_unpacked})







