"""Finds photos that are saved sideways or upside down and records how to turn them.

Some photos lost their rotation tag when they were exported, so every program shows them sideways.
Faces are only found when they are upright, so each photo is tried in 4 directions and the direction
where the face detector is most sure wins. Photos without faces are left alone, use the Rotate
button in the app for those.

    python fix_rotation.py            # writes outputs/rotations.json
    python pipeline.py                # then let the AI describe the photos the right way up
"""
import numpy as np
from insightface.app import FaceAnalysis

import pipeline as p

MARGIN = 0.5  # how much more sure the detector must be before a photo is turned


def face_score(detector, img):
    faces = detector.get(np.array(img)[:, :, ::-1])
    return sum(f.det_score for f in faces if f.det_score >= p.FACE_MIN_SCORE)


def main():
    import onnxruntime  # noqa: F401  (loads before insightface so it finds the CUDA libraries)

    detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=(640, 640))
    files = sorted(f for f in p.DATA.iterdir() if f.suffix.lower() in p.EXTS)
    known = p.rotations()
    changed = []
    for n, path in enumerate(files, 1):
        if path.name in known:
            continue  # already turned, by hand or an earlier run
        img = p.load_image(path, 800)
        scores = {deg: face_score(detector, img.rotate(-deg, expand=True)) for deg in (0, 90, 180, 270)}
        best = max(scores, key=scores.get)
        if best and scores[best] >= scores[0] + MARGIN:
            p.set_rotation(path.name, best)
            changed.append((n, best))
    for n, deg in changed:
        print(f"Photo {n}: turned {deg} degrees clockwise")
    print(f"{len(changed)} photos turned. Run python pipeline.py so the AI describes them the right way up.")


if __name__ == "__main__":
    main()
