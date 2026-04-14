import torch
from munkres import Munkres
from lidar_human_pose_estimation.utils import sensor_utils


def data_to_detections(humans_presence_sensor, humans_distance_sensor, humans_orientation_abs, sensor_angles, sensor_poses):
    mask_humans_presence = humans_presence_sensor > 0
    humans_coordinates = sensor_utils.transform_scan_to_cartesian(
        humans_presence_sensor * humans_distance_sensor, sensor_angles, sensor_poses, alignment_axis="z"
    )

    humans_positions = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [humans_coordinates[i, mask_humans_presence[i]] for i in range(humans_presence_sensor.size(0))],
            dtype=torch.float64,
        ),
        padding=torch.nan,
    )

    humans_orientations = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [humans_orientation_abs[i, mask_humans_presence[i]] for i in range(humans_presence_sensor.size(0))],
            dtype=torch.float64,
        ),
        padding=torch.nan,
    ).unsqueeze(-1)

    return torch.cat([humans_positions, humans_orientations], dim=2)


def match_detections(detections_gt, detections_pred, max_cost: float = 1.5):
    # Ensure that the number of detections are equal
    assert detections_gt.size(0) == detections_pred.size(0), "Number of timestamps must match"

    matching_results = []
    m = Munkres()

    for t in range(detections_gt.size(0)):
        valid_gt_mask = ~torch.isnan(detections_gt[t, :, :]).any(dim=1)
        detection_gt_positions = detections_gt[t, valid_gt_mask, :2]  # (num_gt_detections, 2)
        valid_pred_mask = ~torch.isnan(detections_pred[t, :, :]).any(dim=1)
        detection_pred_positions = detections_pred[t, valid_pred_mask, :2]  # (num_pred_detections, 2)

        num_gt = detection_gt_positions.size(0)
        num_pred = detection_pred_positions.size(0)

        matching_results_ts = {
            "matched_pairs": [],
            "pred_TP_ids": [],
            "pred_FP_ids": [],
            "gt_TP_ids": [],
            "gt_FN_ids": [],
            "num_pred": num_pred,
            "num_gt": num_gt,
        }

        if matching_results_ts["num_pred"] == 0 and matching_results_ts["num_gt"] == 0:
            matching_results.append(matching_results_ts)
            continue

        if matching_results_ts["num_pred"] == 0:  # No predictions, only False Negatives
            matching_results_ts["gt_FN_ids"] = list(range(num_gt))
            matching_results.append(matching_results_ts)
            continue

        if matching_results_ts["num_gt"] == 0:  # No ground truth, only False Positives
            matching_results_ts["pred_FP_ids"] = list(range(num_pred))
            matching_results.append(matching_results_ts)
            continue

        # Compute cost matrix
        cost_matrix = torch.cdist(detection_gt_positions[None, ...], detection_pred_positions[None, ...]).squeeze(0)

        # Apply Hungarian Algorithm (Munkres) for optimal matching
        indexes = m.compute(cost_matrix.detach().cpu().tolist())

        for gt_idx, pred_idx in indexes:
            if cost_matrix[gt_idx, pred_idx] <= max_cost:
                matching_results_ts["matched_pairs"].append((gt_idx, pred_idx, cost_matrix[gt_idx, pred_idx]))
                matching_results_ts["pred_TP_ids"].append(pred_idx)
                matching_results_ts["gt_TP_ids"].append(gt_idx)

        # False Positives
        matching_results_ts["pred_FP_ids"] = list(set(range(num_pred)) - set(matching_results_ts["pred_TP_ids"]))
        # False Negatives
        matching_results_ts["gt_FN_ids"] = list(set(range(num_gt)) - set(matching_results_ts["gt_TP_ids"]))

        matching_results.append(matching_results_ts)

    return matching_results


def compute_matching_metrics(matching_results):
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives

    for matching_results_ts in matching_results:
        total_tp += len(matching_results_ts["pred_TP_ids"])
        total_fp += len(matching_results_ts["pred_FP_ids"])
        total_fn += len(matching_results_ts["gt_FN_ids"])

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "True Positives": total_tp,
        "False Positives": total_fp,
        "False Negatives": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
    }
