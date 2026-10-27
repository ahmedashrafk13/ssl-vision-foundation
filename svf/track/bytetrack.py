"""ByteTrack: Kalman-filtered tracks, two-stage association (high- then low-score detections)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h)."""

    def __init__(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        self.x = np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h, 0, 0, 0, 0], float)
        self.P = np.diag([10, 10, 1e-2, 10, 1e4, 1e4, 1e-4, 1e4]).astype(float)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def _noise(self):
        h = self.x[3]
        q = np.array([h / 20, h / 20, 1e-2, h / 20, h / 160, h / 160, 1e-5, h / 160]) ** 2
        r = np.array([h / 20, h / 20, 1e-1, h / 20]) ** 2
        return np.diag(q), np.diag(r)

    def predict(self):
        Q, _ = self._noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        z = np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h])
        _, R = self._noise()
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self):
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track:
    _next_id = 1

    def __init__(self, box, score, label):
        self.kf, self.score, self.label = KalmanBox(box), score, label
        self.id = Track._next_id
        Track._next_id += 1
        self.lost = 0

    @property
    def box(self):
        return self.kf.box


class ByteTracker:
    def __init__(self, high=0.5, low=0.1, match_iou=0.2, max_lost=30, new_track=0.6):
        self.high, self.low, self.match_iou, self.max_lost, self.new_track = high, low, match_iou, max_lost, new_track
        self.tracks = []

    def _assign(self, tracks, boxes):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = iou_matrix(np.stack([t.box for t in tracks]), boxes)
        r, c = linear_sum_assignment(-iou)
        matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= self.match_iou]
        mt, md = {i for i, _ in matches}, {j for _, j in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def update(self, boxes, scores, labels):
        """boxes (N,4) xyxy, scores (N,), labels (N,) numpy. Returns list of (id, box, score, label)."""
        boxes, scores, labels = map(np.asarray, (boxes, scores, labels))
        for t in self.tracks:
            t.kf.predict()
        hi, lo = scores >= self.high, (scores >= self.low) & (scores < self.high)
        hb, hs, hl = boxes[hi], scores[hi], labels[hi]
        m1, un_t, un_d = self._assign(self.tracks, hb)
        for i, j in m1:
            self.tracks[i].kf.update(hb[j]); self.tracks[i].score = hs[j]; self.tracks[i].lost = 0
        # second association: leftover tracks vs low-score detections (recovers occluded objects)
        rest = [self.tracks[i] for i in un_t]
        m2, un_t2, _ = self._assign(rest, boxes[lo])
        for i, j in m2:
            rest[i].kf.update(boxes[lo][j]); rest[i].lost = 0
        for i in un_t2:
            rest[i].lost += 1
        for j in un_d:
            if hs[j] >= self.new_track:
                self.tracks.append(Track(hb[j], hs[j], hl[j]))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return [(t.id, t.box, t.score, t.label) for t in self.tracks if t.lost == 0]
