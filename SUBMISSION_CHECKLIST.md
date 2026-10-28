# Submission checklist

- [ ] Confirm permission to share source videos, extracted frames and annotations with reviewers.
- [ ] Add `verified_vehicle_labels_104.zip` locally if available (not included in this package).
- [ ] Add the corresponding 104 selected frames locally under `dataset/selected/train_A` ... `train_D` if available.
- [ ] If permitted, share the verified dataset separately with the reviewer; `.gitignore` excludes datasets by default.
- [ ] Add actual evaluation logs/screenshots if available. No metrics were fabricated.
- [ ] Verify paths before running: several scripts expect to run from the project root; `frame_extraction_v2.py` has a hardcoded Windows input path.
- [ ] Review README limitations before submission; final detector and distance-range evaluation are not complete.
- [ ] Check repository privacy and grant reviewer access.

## Suggested local order

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Put source videos in the configured folder and run `frame_extraction_v2.py`.
3. Run `frame_selector_v2.py` to curate frames.
4. Download the pretrained YOLO11m weights locally, then use `yolo_sahi_cvat_10.py` and `yolo_sahi_cvat_remaining.py` as appropriate.
5. Manually review annotations in CVAT; export verified labels.
6. Run `assemble_verified_dataset.py` with the expected local ZIP and selected frames.
7. Run `prepare_train_val.py` for a preliminary same-video split (not independent validation).

**Note:** scripts have not been end-to-end executed here because source videos, verified label archive and image dataset were not provided.
