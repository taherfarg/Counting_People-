"""
Multi-Algorithm Object Tracking Module
Combines multiple tracking algorithms for robust object tracking
"""
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import time
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment

from ..utils.config import TrackingConfig, config


class ObjectTracker:
    """
    Base class for object tracking functionality
    Supports multiple tracking algorithms
    """

    def __init__(self, tracking_config: Optional[TrackingConfig] = None):
        """
        Initialize object tracker

        Args:
            tracking_config: Tracking configuration (uses global config if None)
        """
        self.config = tracking_config or config.tracking
        self.algorithm = self.config.tracking_algorithm
        self.tracker = self._create_tracker()

    def _create_tracker(self):
        """Create appropriate tracker based on algorithm"""
        if self.algorithm == "centroid":
            return CentroidTracker(self.config)
        elif self.algorithm == "sort":
            return SORTTracker(self.config)
        elif self.algorithm == "kalman":
            return KalmanTracker(self.config)
        else:
            raise ValueError(f"Unknown tracking algorithm: {self.algorithm}")

    def update(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries from detector

        Returns:
            List of (object_id, centroid_x, centroid_y) tuples
        """
        return self.tracker.update(detections)

    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        return self.tracker.get_active_tracks()

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track"""
        return self.tracker.get_track_info(track_id)


class CentroidTracker:
    """
    Centroid-based object tracking using Euclidean distance
    Simple and efficient for basic tracking needs
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.last_detected: Dict[int, float] = {}
        self.max_disappeared = config.max_disappeared
        self.disappeared_time_threshold = config.disappeared_time_threshold
        self.centroid_distance_threshold = config.centroid_distance_threshold

    def register(self, centroid: np.ndarray) -> int:
        """Register a new object"""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.last_detected[object_id] = time.time()
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        """Remove an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.last_detected:
            del self.last_detected[object_id]

    def update(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries

        Returns:
            List of (object_id, centroid_x, centroid_y) tuples
        """
        current_time = time.time()

        # If no detections, increment disappeared counter for all objects
        if not detections:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Remove objects that have been missing too long
                if (current_time - self.last_detected[object_id]) > self.disappeared_time_threshold:
                    self.deregister(object_id)
            return [(object_id, *self.objects[object_id]) for object_id in self.objects]

        # Extract centroids from detections
        input_centroids = np.array([detection['center'] for detection in detections])

        # If no existing objects, register all new detections
        if not self.objects:
            for centroid in input_centroids:
                object_id = self.register(centroid)
                self.disappeared[object_id] = 0
        else:
            # Associate detections with existing objects using distance
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find optimal assignment
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Associate detections with objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.last_detected[object_id] = current_time
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched detections (new objects)
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if (current_time - self.last_detected[object_id]) > self.disappeared_time_threshold:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return [(object_id, *self.objects[object_id]) for object_id in self.objects]

    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.objects.keys())

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track"""
        if track_id not in self.objects:
            return None

        current_time = time.time()
        last_seen = current_time - self.last_detected.get(track_id, current_time)

        return {
            'id': track_id,
            'centroid': self.objects[track_id],
            'disappeared_count': self.disappeared.get(track_id, 0),
            'last_seen_seconds': last_seen,
            'is_active': self.disappeared.get(track_id, 0) == 0
        }


class SORTTracker:
    """
    SORT (Simple Online Realtime Tracking) implementation
    More sophisticated tracking with Kalman filtering
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.trackers: List['KalmanBoxTracker'] = []
        self.frame_count = 0
        self.max_age = config.sort_max_age
        self.min_hits = config.sort_min_hits
        self.iou_threshold = config.sort_iou_threshold

    def update(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
        """
        Update SORT tracker with new detections
        """
        self.frame_count += 1

        # Convert detections to SORT format [x1, y1, x2, y2, confidence]
        dets = []
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])

        dets = np.array(dets) if dets else np.empty((0, 5))

        # Update trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # Associate detections with trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Remove unmatched trackers that are too old
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.predict()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                i -= 1
                continue
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        # Return active tracks with centroids
        active_tracks = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()[0]
                centroid_x = (bbox[0] + bbox[2]) / 2
                centroid_y = (bbox[1] + bbox[3]) / 2
                active_tracks.append((trk.id, centroid_x, centroid_y))

        return active_tracks

    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """Associate detections with trackers using IoU"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty((0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def _iou(self, bb_test, bb_gt):
        """Calculate IoU between two bounding boxes"""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        return [trk.id for trk in self.trackers if trk.time_since_update < 1]

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track"""
        for trk in self.trackers:
            if trk.id == track_id:
                bbox = trk.get_state()[0]
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                return {
                    'id': track_id,
                    'centroid': centroid,
                    'bbox': bbox,
                    'hit_streak': trk.hit_streak,
                    'time_since_update': trk.time_since_update
                }
        return None


class KalmanBoxTracker:
    """Kalman filter-based box tracker for SORT"""

    def __init__(self, bbox):
        self.id = KalmanBoxTracker._get_next_id()
        self.boxes = deque(maxlen=30)
        self.boxes.append(bbox)
        self.hits = 1
        self.no_losses = 0
        self.time_since_update = 0

    @staticmethod
    def _get_next_id():
        if not hasattr(KalmanBoxTracker, 'count'):
            KalmanBoxTracker.count = 0
        KalmanBoxTracker.count += 1
        return KalmanBoxTracker.count

    def update(self, bbox):
        self.boxes.append(bbox)
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        return self.boxes[-1]

    def get_state(self):
        if len(self.boxes) == 1:
            return np.array([[self.boxes[0][0], self.boxes[0][1],
                            self.boxes[0][2], self.boxes[0][3], 0, 0, 0, 0]]).reshape((1, 8))
        return np.array(self.boxes[-1]).reshape((1, 4))


class KalmanTracker:
    """
    Basic Kalman filter tracker
    Simplified implementation for basic use cases
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
        """Update Kalman tracker with new detections"""
        # Simple implementation - in practice would use actual Kalman filter
        centroids = []
        for detection in detections:
            center = detection['center']
            # Find closest existing track
            min_distance = float('inf')
            closest_track = None

            for track_id, track_info in self.tracks.items():
                distance = math.hypot(center[0] - track_info['centroid'][0],
                                    center[1] - track_info['centroid'][1])
                if distance < min_distance:
                    min_distance = distance
                    closest_track = track_id

            if closest_track is not None and min_distance < self.config.centroid_distance_threshold:
                # Update existing track
                self.tracks[closest_track]['centroid'] = center
                self.tracks[closest_track]['last_seen'] = time.time()
                centroids.append((closest_track, center[0], center[1]))
            else:
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = {
                    'centroid': center,
                    'created': time.time(),
                    'last_seen': time.time()
                }
                self.next_id += 1
                centroids.append((track_id, center[0], center[1]))

        # Clean up old tracks
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if current_time - track_info['last_seen'] > self.config.disappeared_time_threshold:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return centroids

    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.tracks.keys())

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            return {
                'id': track_id,
                'centroid': track['centroid'],
                'age': time.time() - track['created'],
                'last_seen': time.time() - track['last_seen']
            }
        return None
