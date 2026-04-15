import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Function
from PIL import Image
import cv2
import numpy as np
import os
import streamlit as st

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def build_resnet50_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class AgriAX_DANN(nn.Module):
    def __init__(self, base_model, num_classes):
        super(AgriAX_DANN, self).__init__()
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.class_classifier = nn.Linear(base_model.fc.in_features, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):
        features = torch.flatten(self.feature_extractor(x), 1)
        class_output = self.class_classifier(features)
        return class_output, None

class VisionAnalyzer:
    """비전 모델 추론 및 시각화 전담 클래스"""

    @staticmethod
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 모델 경로가 변경되었으므로 models/ 폴더 기준으로 지정
        model_path = 'models/dann_multicrop_best.pth'

        base_model = build_resnet50_model(num_classes=21)
        model = AgriAX_DANN(base_model=base_model, num_classes=21)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            return model, device
        return None, device

    @staticmethod
    def predict(image, model, device, threshold=0.75):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            class_output, _ = model(img_tensor)
            probabilities = torch.nn.functional.softmax(class_output[0], dim=0)
            max_prob, predicted_idx = torch.max(probabilities, 0)

            if max_prob.item() < threshold:
                return -1, max_prob.item()
        return predicted_idx.item(), max_prob.item()

    @staticmethod
    def generate_gradcam(pil_img, model, device, class_idx):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_img.convert('RGB')).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        gradients = []
        activations = []

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        def fwd_hook(module, input, output):
            activations.append(output.detach())

        target_layer = model.feature_extractor[7]
        handle_fwd = target_layer.register_forward_hook(fwd_hook)
        handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

        model.eval()
        model.zero_grad()
        class_output, _ = model(img_tensor)

        score = class_output[0, class_idx]
        score.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        if not gradients or not activations:
            return pil_img

        grads = gradients[0].cpu().numpy()[0]
        acts = activations[0].cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (pil_img.width, pil_img.height))
        cam -= np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam /= cam_max

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig_np = np.array(pil_img.convert('RGB'))
        superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

        return Image.fromarray(superimposed)
