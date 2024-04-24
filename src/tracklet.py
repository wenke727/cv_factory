import numpy as np
import datetime

class Tracklet:
    def __init__(self, initial_encoding, initial_location, track_id, timestamp=None):
        self.track_id = track_id
        self.face_encodings = [initial_encoding]
        self.face_locations = [initial_location]
        self.timestamps = [timestamp or datetime.datetime.now()]
        self.last_seen = self.timestamps[-1]

    def update(self, new_encoding, new_location, timestamp=None):
        self.face_encodings.append(new_encoding)
        self.face_locations.append(new_location)
        self.timestamps.append(timestamp or datetime.datetime.now())
        self.last_seen = self.timestamps[-1]

    def get_latest(self):
        return {
            'encoding': self.face_encodings[-1],
            'location': self.face_locations[-1],
            'timestamp': self.timestamps[-1]
        }

    def is_active(self, timeout_seconds=300):
        return (datetime.datetime.now() - self.last_seen).total_seconds() < timeout_seconds

    def calculate_similarity(self, encoding):
        # Assuming the use of cosine similarity
        latest_encoding = self.face_encodings[-1]
        cos_sim = np.dot(latest_encoding, encoding) / (np.linalg.norm(latest_encoding) * np.linalg.norm(encoding))
        return cos_sim
