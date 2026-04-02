# Object Tracking by Detection

This repository focuses on Object Tracking by Detection. The tracking algorithms can be categorized into five generations based on their evolution and focus areas:

## Generation 1 – Classic (Speed) 

* **SORT (2016)**: Uses a Kalman filter for motion prediction + IoU-based association via the Hungarian Algorithm. Very fast (~260 FPS), but barely any Re-ID capabilities.
* **IoU Tracker (2017)**: Even simpler – purely based on bounding box overlap. Extremely fast but fails under occlusion.
* **Centroid Tracker**: Associates tracks using the Euclidean distance between bounding box centers.

## Generation 2 – Appearance / Re-ID

* **DeepSORT (2017)**: Extends SORT with a CNN embedding for visual similarity, making it tolerate occlusions much better.
* **StrongSORT (2022)**: An upgrade to DeepSORT using EMA embeddings (for smoother feature updates), OSNet Re-ID, and AFLink post-processing.
* **BoT-SORT (2022)**: Adds Global Motion Compensation (GMC), which compensates for camera motion before the matching step.

## Generation 3 – High Performance

* **ByteTrack (2022)**: Core idea – even low-confidence detections are used for association (no hard confidence cutoff). Reached SOTA on MOTChallenge.
* **OC-SORT (2022)**: Observation-Centric SORT. Corrects Kalman drift during occlusions through reactivation based on real observations.
* **MotionTrack (2023)**: Optimized for fast, non-linear movements.

## Generation 4 – Transformer / End-to-End

* **TrackFormer (2021)**: DETR-based model; handles both detection and tracking in a single network.
* **MOTR (2021)**: Track Queries propagate across frames, eliminating the need for a separate Re-ID model.
* **MeMOT (2022)**: Features a long-term memory mechanism designed for recurring objects.

## Generation 5 – Specialized / Hybrid

* **SparseTrack (2023)**: Uses a pseudo-depth map for improved occlusion handling.
* **Deep OC-SORT**: Combines OC-SORT with appearance costs for a hybrid association approach.
* **Hybrid SORT (2023)**: Combines IoU, pose keypoints, and appearance features.