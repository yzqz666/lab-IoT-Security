import os
import tempfile
import numpy as np
import cv2
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

DB_IMAGES = ["Hu.jpg", "Peng.jpg"]
TEST_IMAGES = [f for f in os.listdir(".") if f.lower().startswith("test") 
and f.lower().endswith(('.jpg'))]
DETECTOR = MTCNN()
MODEL_NAME = 'ArcFace'
DETECTOR_BACKEND = 'mtcnn' 


def load_img_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def detect_faces(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = DETECTOR.detect_faces(img_rgb)
    return faces


def crop_box(img_bgr, box, expand_ratio=0.0):
    x, y, w, h = box
    x1 = max(0, int(x - w * expand_ratio))
    y1 = max(0, int(y - h * expand_ratio))
    x2 = min(img_bgr.shape[1], int(x + w + w * expand_ratio))
    y2 = min(img_bgr.shape[0], int(y + h + h * expand_ratio))
    return img_bgr[y1:y2, x1:x2]


def save_temp_image(img_bgr):
    fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    cv2.imwrite(tmp_path, img_bgr)
    return tmp_path


def robust_represent(model, img_path_or_array, detector_backend='mtcnn'):
    cleanup = False
    if isinstance(img_path_or_array, np.ndarray):
        tmp = save_temp_image(img_path_or_array)
        img_path = tmp
        cleanup = True
    else:
        img_path = img_path_or_array

    rep = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, 
    detector_backend=detector_backend, enforce_detection=True)

    if cleanup:
        try:
            os.remove(img_path)
        except Exception:
            pass
    emb = None
    if isinstance(rep, dict):
        if 'embedding' in rep:
            emb = np.array(rep['embedding'], dtype=np.float32)
        else:
            vals = list(rep.values())
            emb = np.array(vals[0], dtype=np.float32)
    elif isinstance(rep, list):
        if len(rep) == 0:
            raise ValueError('未提取到特征向量')
        first = rep[0]
        if isinstance(first, dict) and 'embedding' in first:
            emb = np.array(first['embedding'], dtype=np.float32)
        elif isinstance(first, (list, np.ndarray)):
            emb = np.array(first, dtype=np.float32)
        else:
            emb = np.array(first, dtype=np.float32)
    elif isinstance(rep, (np.ndarray, list)):
        emb = np.array(rep, dtype=np.float32)

    if emb is None:
        raise RuntimeError(f"无法解析 DeepFace.represent 返回值: {type(rep)}")

    # 确保为 1-d
    emb = emb.reshape(-1)
    return emb


def draw_boxes_and_labels(img_bgr, faces, labels=None):
    out = img_bgr.copy()
    if labels is None:
        labels = [None] * len(faces)
    for (f, label) in zip(faces, labels):
        x, y, w, h = f['box']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - int(th * 1.6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def compute_db_embeddings(model, db_image_paths):
    db = {}
    for p in db_image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"库图片不存在: {p}")
        img = load_img_bgr(p)
        faces = detect_faces(img)
        if len(faces) == 0:
            raise RuntimeError(f"在库图片中未检测到人脸: {p}")
        faces_sorted = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        face_crop = crop_box(img, faces_sorted[0]['box'], expand_ratio=0.3)
        emb = robust_represent(model, face_crop, detector_backend=DETECTOR_BACKEND)
        db[p] = {
            'embedding': emb,
            'face_box': faces_sorted[0]['box']
        }
        print(f"已为库图片 {p} 提取特征向量 (len={len(emb)})")
    return db


def match_face(emb, db_embeddings):
    sims = []
    for p, info in db_embeddings.items():
        db_emb = info['embedding']
        sim = cosine_similarity(emb.reshape(1, -1), db_emb.reshape(1, -1))[0][0]
        sims.append((p, float(sim)))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims_sorted


def main():
    print("正在加载 ArcFace 模型")
    model = DeepFace.build_model(MODEL_NAME)
    print("模型加载完成")

    print("为库图片提取特征向量...")
    db = compute_db_embeddings(model, DB_IMAGES)

    if len(TEST_IMAGES) == 0:
        print("未在当前目录发现任何 test*.jpg/png 图片。请把测试图片命名为 test1.jpg, test2.jpg 等。")
        return

    for tpath in TEST_IMAGES:
        print('\n' + '=' * 60)
        print(f"处理测试图片: {tpath}")
        img = load_img_bgr(tpath)
        faces = detect_faces(img)
        print(f"检测到 {len(faces)} 张人脸")

        labels = []
        for idx, f in enumerate(faces):
            face_crop = crop_box(img, f['box'], expand_ratio=0.2)
            emb = robust_represent(model, face_crop, detector_backend=DETECTOR_BACKEND)
            sims_sorted = match_face(emb, db)
            best = sims_sorted[0]
            label = f"{os.path.basename(best[0])}: {best[1]:.4f}"
            labels.append(label)


            print(f" 人脸 #{idx+1} 最佳匹配: {best[0]} 相似度={best[1]:.4f}")
            print("  Top 列表:")
            for p, s in sims_sorted[:1]:
                print(f"    {os.path.basename(p)} -> {s:.4f}")

        out = draw_boxes_and_labels(img, faces, labels)
        out_name = os.path.splitext(tpath)[0] + '_boxed.jpg'
        cv2.imwrite(out_name, out)
        print(f"已生成标注图片: {out_name}")
        try:
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(out_rgb)
            plt.axis('off')
            plt.title(f"{tpath} - 匹配结果")
            plt.show()
        except Exception as e:
            print("展示图像时出错（可能是无显示环境）:", e)


if __name__ == '__main__':
    main()
