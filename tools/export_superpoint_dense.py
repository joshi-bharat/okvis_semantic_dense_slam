#!/usr/bin/env python3
"""Export SuperPoint (Magic Leap superpoint_v1.pth) to ONNX with DENSE outputs for OKVIS2-X.

The OKVIS SuperPoint frontend does non-maximum suppression and descriptor sampling in C++, so the
network must stop before them. The exported graph has

    input   image        float32 [1, 1, H, W]        greyscale in [0, 1]
    output  scores       float32 [1, H, W]           keypoint probability per pixel
    output  descriptors  float32 [1, 256, H/8, W/8]  L2-normalised coarse descriptor grid

H and W must be multiples of 8 and equal to the size the frontend feeds (the image size, or
camera_parameters/image_size when resizing).

Usage:
    python3 tools/export_superpoint_dense.py superpoint_v1.pth models/superpoint_1024.onnx \
        --height 1024 --width 1024
    trtexec --onnx=models/superpoint_1024.onnx --saveEngine=models/superpoint_1024.engine --fp16
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointDense(nn.Module):
    """Magic Leap SuperPointNet, ending at the dense score map and coarse descriptors."""

    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, d1, 1, 1, 0)

    def forward(self, image):
        x = F.relu(self.conv1a(image))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        # 65 channels per 8x8 cell: 64 pixel positions plus a "no keypoint" dustbin.
        semi = self.convPb(F.relu(self.convPa(x)))
        probabilities = F.softmax(semi, dim=1)[:, :64]
        scores = F.pixel_shuffle(probabilities, 8)[:, 0]  # [1, H, W]

        descriptors = self.convDb(F.relu(self.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)  # [1, 256, H/8, W/8]
        return scores, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", help="superpoint_v1.pth")
    parser.add_argument("output", help="output .onnx")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    if args.height % 8 or args.width % 8:
        parser.error("height and width must be multiples of 8")

    model = SuperPointDense()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.rand(1, 1, args.height, args.width)
    torch.onnx.export(model, dummy, args.output, input_names=["image"],
                      output_names=["scores", "descriptors"], opset_version=17)
    print(f"wrote {args.output}: image [1,1,{args.height},{args.width}] -> "
          f"scores [1,{args.height},{args.width}], "
          f"descriptors [1,256,{args.height // 8},{args.width // 8}]")


if __name__ == "__main__":
    main()
