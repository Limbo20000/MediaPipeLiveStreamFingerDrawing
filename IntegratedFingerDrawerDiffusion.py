import time
import cv2
import mediapipe as mp
import numpy as np
import random
import torch
import gradio as gr
import controlnet_aux
import PIL.Image
import time
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

print(torch.cuda.is_available())

MAX_NUM_IMAGES = 5
DEFAULT_NUM_IMAGES = 3
MAX_IMAGE_RESOLUTION = 768
DEFAULT_IMAGE_RESOLUTION = 768

MAX_SEED = np.iinfo(np.int32).max


def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    # area interpolation for downsizing, lanczos for upsizing
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img


device = "cuda"
task_name = "scribble"

# using srcibble-based ControlNet with Stable Diffusion 1.5
base_model_id = "runwayml/stable-diffusion-v1-5"
model_id = "lllyasviel/control_v11p_sd15_scribble"

# instantiate the model and its pipeline form diffusers
controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)


# image generation from scribble input
# using torch.inference_mode to disable gradient tracking
@torch.inference_mode()
def process_scribble_interactive(
        image_and_mask: dict[str, np.ndarray],
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
) -> list[PIL.Image.Image]:
    if image_and_mask is None:
        raise ValueError
    if not image_resolution:
        image_resolution = 768
    if num_images > MAX_NUM_IMAGES:
        raise ValueError

    if saved_image is not None:
        image = saved_image  # Use the saved image
    else:
        image = image_and_mask["mask"]

    image = controlnet_aux.util.HWC3(image)
    image = resize_image(image, resolution=image_resolution)
    control_image = PIL.Image.fromarray(image)

    if not prompt:
        prompt = additional_prompt
    else:
        prompt = f"{prompt}, {additional_prompt}"

    generator = torch.Generator().manual_seed(seed)
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_inference_steps=num_steps,
        generator=generator,
        image=control_image,
    ).images
    return [control_image] + results


# random seed utility
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


saved_image = None
last_canvas = np.full((DEFAULT_IMAGE_RESOLUTION, DEFAULT_IMAGE_RESOLUTION, 3), 255, dtype=np.uint8)

mp_hands = mp.solutions.hands


def draw_index_fingertip(image):
    global last_canvas

    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return last_canvas

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as hands:
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is present

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if 'prev_x' in globals() and 'prev_y' in globals():
                cv2.line(last_canvas, (prev_x, prev_y), (x, y), (0, 255, 120), 2)

            globals()['prev_x'], globals()['prev_y'] = x, y

    return last_canvas


def clear_canvas():
    global last_canvas
    last_canvas = np.full((DEFAULT_IMAGE_RESOLUTION, DEFAULT_IMAGE_RESOLUTION, 3), 255, dtype=np.uint8)
    return last_canvas


def save_image():
    global saved_image, last_canvas
    saved_image = np.copy(last_canvas)


# create Gradio-based user interface
# based on: https://huggingface.co/spaces/hysts/ControlNet
def create_demo(process):
    resolution_choices = [256, 512, 768]
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tabs(["Sketch with your fingers"]):
            with gr.Tab("Sketch with your fingers"):
                with gr.Row():
                    image_input = gr.Image(
                        shape=(int(DEFAULT_IMAGE_RESOLUTION / 2), int(DEFAULT_IMAGE_RESOLUTION / 2)),
                        source="webcam", streaming=True, type="numpy", label="Livestream Feed")
                    image_output = gr.Image(
                        shape=(int(DEFAULT_IMAGE_RESOLUTION / 2), int(DEFAULT_IMAGE_RESOLUTION / 2)), type="numpy",
                        label="Canvas")
                prompt = gr.Textbox(label="Prompt")
                with gr.Row():
                    a_prompt = gr.Textbox(
                        label="Additional prompt", value=""
                    )
                    n_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="",
                    )
                    image_resolution = gr.Dropdown(
                        label="Image resolution",
                        choices=resolution_choices,
                        default=DEFAULT_IMAGE_RESOLUTION,
                    )

                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                num_samples = gr.Radio(
                    label="Number of images",
                    choices=list(range(1, MAX_NUM_IMAGES + 1)),
                    value=DEFAULT_NUM_IMAGES,
                )

                num_steps = gr.Slider(
                    label="Number of steps", minimum=1, maximum=100, value=1, step=1
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1
                )

                with gr.Row():
                    clear_button = gr.Button("Clear Sketch")
                    save_button = gr.Button("Save Sketch")
                    clear_button.click(fn=clear_canvas, inputs=[], outputs=image_output)
                    save_button.click(fn=save_image, inputs=[], outputs=[])

        run_button = gr.Button("Run", variant="primary")

        result = gr.Gallery(
            label="Output", show_label=False, columns=2
        )

        image_input.stream(fn=draw_index_fingertip, inputs=image_input, outputs=image_output)
        clear_button.click(fn=clear_canvas, inputs=[], outputs=image_output)
        save_button.click(fn=save_image, inputs=[], outputs=[])

        inputs = [
            image_input,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            num_steps,
            guidance_scale,
            seed,
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
        )
    return demo


demo = create_demo(process_scribble_interactive)
demo.queue().launch(debug=True, share=True)
