# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys, os
sys.path.insert(0, os.getcwd())

import mmcv
import torch
import imageio.v2 as imageio

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

import plotly.graph_objects as go
import plotly.offline as pyo


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default="./iter_80000.pth")
    parser.add_argument('--video_dir', help='path to input image file')
    parser.add_argument('--output_dir', help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def get_seg_results(seg_model, frames, device, edge_blur=16):

    seg_results = [inference_segmentor(seg_model, frame)[0] for frame in frames]
    seg_results = [torch.tensor(seg_result, dtype=torch.int, device=device) for seg_result in seg_results]
    masks = [(seg_result == 12).to(torch.float) for seg_result in seg_results]

    blurred_masks = [torch.nn.functional.interpolate(mask.view(1, 1, mask.shape[0], mask.shape[1]), scale_factor=1 / edge_blur, mode="area") for mask in masks]
    blurred_masks = [torch.nn.functional.interpolate(blurred_masks[i], size=frames[i].shape[:2], mode="bicubic").clamp_(0., 1.) for i in range(len(frames))]
    blurred_masks = [mask.view(mask.shape[2], mask.shape[3], 1) for mask in blurred_masks]

    frames = [torch.tensor(frame, dtype=torch.float, device=device) for frame in frames]
    frames = [frames[i] * blurred_masks[i] + torch.mean(frames[i]) * (1. - blurred_masks[i]) for i in range(len(frames))]
    frames = [frame.to(torch.uint8).cpu().numpy() for frame in frames]

    return frames

def analyze_video(model, seg_model, video_path, output_dir, attribute_list, n_samples=20, vis=False):

    video = imageio.get_reader(video_path)
    meta = video.get_meta_data()
    length = int(meta["fps"] * meta["duration"])
    indices = list(range(0, length, length//n_samples))
    frames = [video.get_data(_) for _ in indices]
    frames = get_seg_results(seg_model, frames, torch.device("cuda:0"))

    outputs = []
    for frame in tqdm(frames):
        
        imageio.imwrite("/tmp/tmp_image.png", frame)
        output, attributes = restoration_inference(model, "/tmp/tmp_image.png", return_attributes=True)
        attributes = attributes.float().detach().cpu().numpy()[0]
        # attributes = [*attributes, attributes[0]]
        outputs.append(attributes)

    outputs = np.array(outputs).mean(axis=0) # [7]
    outputs[-1] = (outputs[0] + outputs[2] + 1 - outputs[3]) / 3

    if vis:
        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=[*outputs, outputs[0]],
                    theta=[*attribute_list, attribute_list[0]],
                ),
            ],
            layout=go.Layout(
                title=go.layout.Title(text=f'Q={outputs[0]:.2f}, S={outputs[2]:.2f}, N={outputs[3]:.2f}, Q+S-N={outputs[-1]:.2f}'),
                polar=dict(
                    radialaxis=dict(range=[0, 1], showticklabels=True, ticks=''),
                    angularaxis=dict(showticklabels=True, ticks='')
                ),
                showlegend=True,
            )
        )
        fig.update_xaxes(tickfont_family="Arial Black")
        fig.write_image(os.path.join(output_dir, f"{os.path.basename(video_path)}.png"), engine="kaleido")

    imageio.mimwrite(os.path.join(output_dir, f"{os.path.basename(video_path)}"), frames, quality=9, fps=4)
    np.save(os.path.join(output_dir, f"{os.path.basename(video_path)}.npy"), outputs)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seg_config = "mmseg_configs/vit/upernet_deit-b16_512x512_160k_ade20k.py"
    seg_checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_160k_ade20k/upernet_deit-b16_512x512_160k_ade20k_20210621_180100-828705d7.pth"

    model = init_model(args.config, args.checkpoint, device=torch.device('cuda', args.device))
    seg_model = init_segmentor(seg_config, seg_checkpoint, device=torch.device("cuda:0"))

    attribute_list = ['Quality', 'Brightness', 'Sharpness', 'Noisiness', 'Colorfulness', 'Q+S-N']
    # attribute_list = [*attribute_list, attribute_list[0]]

    videos = [_ for _ in os.listdir(args.video_dir) if _.endswith(".mp4") and "vis" not in _]
    video_paths = [os.path.join(args.video_dir, _) for _ in videos]

    for path in tqdm(video_paths):
        analyze_video(model, seg_model, path, args.output_dir, attribute_list)
        # try:
        #     analyze_video(model, path, args.output_dir, attribute_list)
        # except Exception as e:
        #     print(f"error occured: {e}")




if __name__ == '__main__':
    main()
