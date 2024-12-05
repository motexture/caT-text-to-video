import argparse
import torch
import random
import subprocess
import json
import numpy as np
import cv2
import os
import gradio as gr

from models.caT import caT
from pipeline.caT import caTPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class VideoGenerator:
    def __init__(self):
        self.device = "cuda:1"
        self.pipeline = self.initialize_pipeline("caT4/train_2024-12-05T16-27-57/validated-4000")
        self.stacked_latents = None
        self.previous_latents = None
        self.generated = False
        self.video_path = "concatenated_output.mp4"
        self.encoded_path = "concatenated_output_encoded.mp4"

    def initialize_pipeline(self, model):  
        print("Loading pipeline...")
        pipeline = caTPipeline.from_pretrained(pretrained_model_name_or_path=model).to(self.device)
        #pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", algorithm_type="sde-dpmsolver++")
        pipeline.vae.enable_slicing()

        return pipeline
    
    def match_histogram(self, source, target):
        """Adjust the color histogram of the target to match the source."""
        mean_src = source.mean(dim=(2, 3, 4), keepdim=True)
        std_src = source.std(dim=(2, 3, 4), keepdim=True)
        
        mean_tgt = target.mean(dim=(2, 3, 4), keepdim=True)
        std_tgt = target.std(dim=(2, 3, 4), keepdim=True)
        
        adjusted_target = (target - mean_tgt) * (std_src / (std_tgt + 1e-5)) + mean_src
        return adjusted_target

    def generate(self, prompt, negative_prompt, interpolation_strength, width, height, num_frames, num_inference_steps, guidance_scale, fps):
        if self.seed != -1:
            set_seed(self.seed)

        with torch.no_grad(), torch.autocast(self.device):
            latents = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                device=self.device,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                previous_latents=self.previous_latents,
                interpolation_strength=interpolation_strength
            )
            self.stacked_latents = torch.cat((self.stacked_latents, self.match_histogram(self.stacked_latents, latents)), dim=2) if self.stacked_latents is not None else latents
            self.stacked_latents = self.normalize_latents(self.stacked_latents)
            self.previous_latents = latents

            self.save_video(self.decode(self.stacked_latents), self.video_path, fps)
            '''
            try:
                encode_video(self.video_path, self.encoded_path, get_video_height(self.video_path))
                os.remove(self.video_path)
            except:
                pass
            '''

            return self.video_path
    
    def normalize_latents(self, latents):
        mean = latents.mean()
        std = latents.std()
        normalized_latents = (latents - mean) / (std + 1e-8)
        return normalized_latents
    
    def denormalize(self, normalized_tensor):
        if normalized_tensor.is_cuda:
            normalized_tensor = normalized_tensor.cpu()
        
        if normalized_tensor.dim() == 5:
            normalized_tensor = normalized_tensor.squeeze(0)
            
        denormalized = (normalized_tensor + 1.0) * 127.5
        denormalized = torch.clamp(denormalized, 0, 255)
        
        uint8_tensor = denormalized.to(torch.uint8)
        uint8_numpy = uint8_tensor.permute(1, 2, 3, 0).numpy()
        
        return uint8_numpy

    def save_video(self, normalized_tensor, output_path, fps=30):
        denormalized_frames = self.denormalize(normalized_tensor)
        height, width = denormalized_frames.shape[1:3]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in denormalized_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()

    def decode(self, latents):
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents

        batch, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)
        image = self.pipeline.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )

        return video.float()        

    def reset_and_generate_initial(self, prompt, negative_prompt, interpolation_strength, width, height, num_frames, num_inference_steps, guidance_scale, fps, seed):
        self.stacked_latents = None
        self.previous_latents = None
        self.generated = False
        self.seed = seed

        return self.generate(prompt, negative_prompt, interpolation_strength, width, height, num_frames, num_inference_steps, guidance_scale, fps)

video_gen = VideoGenerator()
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Darth Vader is surfing on the ocean")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            interpolation_strength = gr.Slider(label="Interpolation Strength", minimum=0.0, maximum=1.0, step=0.1, value=0.5)

            width = gr.Slider(label="Width", minimum=64, maximum=2048, step=64, value=256)
            height = gr.Slider(label="Height", minimum=64, maximum=2048, step=64, value=256)

            num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=60, step=1, value=8)
            num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=150, step=1, value=30)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.1, value=12)
            fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=6)

            seed = gr.Slider(label="Seed", minimum=-1, maximum=10000, step=1, value=-1)
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            generate_initial_button = gr.Button("Generate Initial Video")
            extend_button = gr.Button("Extend Video")

        generate_initial_button.click(
            video_gen.reset_and_generate_initial, 
            inputs=[prompt, negative_prompt, interpolation_strength, width, height, num_frames, num_inference_steps, guidance_scale, fps, seed],
            outputs=video_output
        )

        extend_button.click(
            video_gen.generate, 
            inputs=[prompt, negative_prompt, interpolation_strength, width, height, num_frames, num_inference_steps, guidance_scale, fps],
            outputs=video_output
        )

if __name__ == "__main__":
    iface.launch()