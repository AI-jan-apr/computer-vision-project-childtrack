# Feature Extraction باستخدام نموذج ReID
# حفظ الـ Feature Vector لمقارنة لاحقاً
# مقارنة الـ Feature Vectors لتحديد إذا كان الشخص هو نفسه في فريمات مختلفة
import cv2
import numpy as np
import torch
import torchreid


class ReID:
    def __init__(self):
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()

        _, self.transform = torchreid.data.transforms.build_transforms(
            height=256,
            width=128
        )

        self.database = {}
        self.next_id = 0

    def extract(self, img):
        if img is None or img.size == 0:
            return None

        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img = Image.fromarray(img)

        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            feat = self.model(img)

        emb = feat.cpu().numpy().flatten()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def match(self, embedding, threshold=0.75):
        if embedding is None:
            return None

        best_id = None
        best_score = -1

        for pid, emb in self.database.items():
            score = self.cosine(embedding, emb)
            if score > best_score:
                best_score = score
                best_id = pid

        if best_score > threshold:
            return best_id

        self.database[self.next_id] = embedding
        self.next_id += 1
        return self.next_id - 1