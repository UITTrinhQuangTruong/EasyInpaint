import os
import sys
import argparse
from PIL import Image
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import tempfile
# from omegaconf import OmegaConf
# from sam_segment import predict_masks_with_sam
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from utils import dilate_mask 
from simple_segment import build_simpleclick_model, segment_img_with_builded_simpleclick
from simple_segment import Clicker

to_tensor = ToTensor()


def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def reset(*args):
    return [None for _ in args]

def process_image_click(origin_image, image_nd, clicked_points, image_resolution, click_mask, pre_mask, is_dilate_mask, kernel_size, evt: gr.SelectData):
    x, y = evt.index
    height, width = origin_image.shape[:2]
    positive = (click_mask[y, x, 0] <= 127) if click_mask is not None else True
    new_x = image_resolution * x / width
    new_y = image_resolution * y / height
    clicked_points.append(
        Clicker(new_x, new_y, positive, len(clicked_points) + 1))

    pre_mask = segment_img_with_builded_simpleclick(
        model["seg"], image_nd, pre_mask, clicked_points, [image_resolution]*2, device=device).cpu()

    pred = torch.nn.functional.interpolate(pre_mask, size=(height, width),
                                           mode='bilinear', align_corners=True)
    pred = pred.permute(0, 2, 3, 1).squeeze(0).data.numpy()
    # if click_mask is not None:
    #     pred *= click_mask 
    # else:
    pred *= 255
    pred[pred > 127] = 255
    pred[pred <= 127] = 0
    click_mask = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if is_dilate_mask:
        click_mask = dilate_mask(click_mask, kernel_size)
    for click in clicked_points:
        # Set the circle color based on the label
        color = (255, 0, 0) if click.is_positive else (0, 0, 255)

        # Draw the circle
        origin_image = cv2.circle(origin_image, (x, y), 3, color, -1)

    # Set the opacity for the mask_image and origin_image
    opacity_mask = 0.75
    opacity_edited = 1.0

    # Combine the origin_image and the mask_image using cv2.addWeighted()
    overlay_image = cv2.addWeighted(
        origin_image,
        opacity_edited,
        (click_mask *
            np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        opacity_mask,
        0,)
    return overlay_image, clicked_points, click_mask, pre_mask


def image_upload(image, resolution):
    if isinstance(image, Image.Image):
        image = np.array(image)
    origin_image = image.copy()

    image = to_tensor(image).unsqueeze(0)
    image = F.interpolate(image, size=(resolution, resolution),
                          mode='bilinear', align_corners=True)

    return origin_image, image, [], None, None


def get_inpainted_img(image, mask, image_resolution):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image, mask, lama_config, device=device)
    return img_inpainted


def change_input_img(input_image, origin_image, img_rm_with_mask):
    if img_rm_with_mask is not None:
        origin_image = img_rm_with_mask
        input_image = img_rm_with_mask 
    return input_image, origin_image, [], None, None

# get args
# build models
model = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
# point_ckpt = './pretrained_models/vit_huge.pth'
point_ckpt = '/media/truongtq/data/workspace/labeling/smartlabel/smartlabel/weights/vit_huge.pth'
model["seg"] = build_simpleclick_model(point_ckpt, device=device)


# build the lama model
lama_config = './lama/configs/prediction/default.yaml'
lama_ckpt = './pretrained_models/big-lama'
lama_ckpt = '/media/truongtq/data/workspace/vid2vid/3DSwap/pretrained_models/big-lama'
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100, 50)
css = """
.container {
    height: 100vh;
}
"""
with gr.Blocks(css=css) as demo:
    image_resolution = gr.State(448)
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    image_np = gr.State(None)
    click_mask = gr.State(None)
    # features = gr.State(None)
    pre_mask = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row(elem_classes=[]):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image")
            with gr.Row():
                # img = gr.Image(label="Input Image")
                source_image_click = gr.Image(
                    label="Input Image", show_label=False, elem_id="img2maskimg", 
                    source="upload", interactive=True, type="numpy", height='auto'
                )

        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            with gr.Row():
                with gr.Column():
                    is_dilate_mask = gr.Checkbox(value=True, label="Dilate Mask")
                    kernel_size = gr.Slider(
                        label="Kernel Size",
                        minimum=3,
                        maximum=29,
                        value=15,
                        step=2,
                    )
            with gr.Row():
                undo = gr.Button("Undo", variant="stop")
                done = gr.Button("Apply Class", variant="primary")

            lama = gr.Button("Inpaint Image", variant="primary")
            apply = gr.Button("Apply inpainted image", variant="secondary")

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Removed with Mask")
            with gr.Row():
                img_rm_with_mask = gr.Image(
                    type="numpy", label="Image Removed with Mask", height="auto")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Mask")
            with gr.Row():
                click_mask = gr.Image(type="numpy", label="Click Mask")
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Replace Anything with Mask")
            with gr.Row():
                img_replace_with_mask = gr.Image(
                    type="numpy", label="Image Replace Anything with Mask")

    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[origin_image, image_np, clicked_points, click_mask, pre_mask]
    )
    source_image_click.select(
        process_image_click,
        inputs=[origin_image, image_np, clicked_points, 
                image_resolution, click_mask,
                pre_mask, is_dilate_mask, kernel_size],
        outputs=[source_image_click, clicked_points, click_mask, pre_mask]
    )
    apply.click(
        change_input_img,
        [source_image_click, origin_image, img_rm_with_mask],
        [source_image_click, origin_image, clicked_points, click_mask, pre_mask]
    )

    lama.click(
        get_inpainted_img,
        [origin_image, click_mask, image_resolution],
        [img_rm_with_mask]
    )


if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name='0.0.0.0', share=False, debug=True)
