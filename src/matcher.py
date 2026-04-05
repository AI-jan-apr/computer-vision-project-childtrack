# مقارنة Embeddings الخارجين مع سجلات الدخول
# حساب Cosine Similarity
# تحديد هل هذا نفس الشخص اللي دخل
# ربط كل خارج بـ Group ID معين
# اتخاذ قرار ان الطفل  موجود او لا 

<<<<<<< HEAD
#السيناريوهات
=======
import numpy as np


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class Matcher:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.database = {}   # person_id -> embedding
        self.next_id = 0

    def match(self, embedding):
        best_id = None
        best_score = 0

        for pid, emb in self.database.items():
            score = cosine_similarity(embedding, emb)

            if score > best_score:
                best_score = score
                best_id = pid

        if best_score > self.threshold:
            return best_id

        # new person
        self.database[self.next_id] = embedding
        self.next_id += 1

        return self.next_id - 1
>>>>>>> 3899973241b8265b0b42e9e637e99327a05673c1
