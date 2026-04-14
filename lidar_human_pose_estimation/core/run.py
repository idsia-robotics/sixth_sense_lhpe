import pickle
import numpy as np
import tqdm
import torch
import argparse
from numbers import Number
from lidar_human_pose_estimation.core.model import LHPELossFunction
from typing import Optional, Dict, Callable
from lidar_human_pose_estimation.core.metrics import angular_difference, circular_pearson, distance_mae, detection_and_pose_metrics, orientation_absolute_error, distance_ape, presence_average_iou
from lidar_human_pose_estimation.utils import post_processing_utils
from scipy.stats import circmean

def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    training_config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    selected_loss_function: Callable = LHPELossFunction().loss_function,
    testing = False
) -> Dict[str, Number]:
    training_mode = optimizer is not None

    if not args.quiet:
        dataloader = tqdm.tqdm(dataloader, total=len(dataloader), ncols=100)

    epoch_loss = dict(loss=[])
    metrics = {
        'orientation/mae' : [],
        'orientation/symmetric_mae' : [],
        'distance/mape' : [],
        'presence/avg_iou' : [],
        'orientation/circular_pearson' : []
    }


    if testing:
        metrics['presence/tpr'] = []
        metrics['presence/fpr'] = []
        metrics['presence/fnr'] = []
        metrics['presence/position_error'] = []
        metrics['presence/auc'] = []


    for batch in dataloader:
        if training_mode:
            optimizer.zero_grad()

        batch = {k: v.to(args.device) for k, v in batch.items()}
        pred = model(batch["scan_virtual_history"])
        losses = selected_loss_function(pred=pred, gt=batch)
        loss = torch.stack(list(losses.values()), dim=0).sum()

        fov_mask = batch["camera_fov_mask"] > 0

        if training_config["train_configs"]["loss_function"]["bidirection"]:
            pred["predicted_direction"] = torch.atan2(pred["sine"], pred["cosine"]) / 2.0

            if training_config["train_configs"]["loss_function"]["verse"]:
                pred["predicted_orientation"] = (
                    pred["verse"].sign().clamp(-1, 0)
                    * torch.pi
                    + pred["predicted_direction"]
                )
            else:
                pred["predicted_orientation"] = pred["predicted_direction"]
        else:
            pred["predicted_orientation"] = torch.atan2(pred["sine"], pred["cosine"])


        with torch.no_grad():
            human_presence_mask = batch['humans_presence_sensor'][fov_mask] > 0
            metrics["orientation/symmetric_mae"].append(
                orientation_absolute_error(
                    pred=pred["predicted_orientation"][fov_mask],
                    true=batch["humans_relative_bearing_sensor"][fov_mask],
                    human_presence_mask=human_presence_mask,
                    symmetric=True,
                )
            )

            metrics["orientation/mae"].append(
                orientation_absolute_error(
                    pred=pred["predicted_orientation"][fov_mask],
                    true=batch["humans_relative_bearing_sensor"][fov_mask],
                    human_presence_mask=human_presence_mask,
                    symmetric=False,
                )
            )

            metrics['orientation/circular_pearson'].append(
                circular_pearson(
                    pred['predicted_orientation'][fov_mask][human_presence_mask],
                    batch['humans_relative_bearing_sensor'][fov_mask][human_presence_mask],
                )
            )
            metrics['distance/mape'].append(
                distance_ape(
                    true=batch["humans_distance_sensor"][fov_mask],
                    pred=pred["distance"][fov_mask],
                    human_presence_mask=human_presence_mask,
                )
            )

            metrics['presence/avg_iou'].append(
                presence_average_iou(
                    presence_pred_raw=pred['presence'], presence_gt=batch['humans_presence_sensor'], fov_mask=fov_mask
                )
            )

            if testing:
                nms = post_processing_utils.CircularTensorNMS(threshold=0.95, min_peak_distance=5)
                nms_pred = nms.connected_components_nms(pred)

                detection_metrics = detection_and_pose_metrics(
                    presence_pred_raw=nms_pred['presence'],
                    presence_true=batch['humans_presence_sensor'],
                    dist_pred=pred['distance'],
                    dist_true=batch['humans_distance_sensor']
                )
                metrics['presence/fpr'].append(detection_metrics['fpr'][None, ...])
                metrics['presence/tpr'].append(detection_metrics['tpr'][None, ...])
                metrics['presence/fnr'].append(detection_metrics['fnr'][None, ...])
                metrics['presence/position_error'].append(detection_metrics['matching_cost'][None, ...])
                metrics['presence/auc'].append(detection_metrics['auc'][None, ...])

        if training_mode:
            loss.backward()
            optimizer.step()

        epoch_loss["loss"].append(loss.cpu())

        for l, v in losses.items():
            if l in epoch_loss:
                epoch_loss[l].append(v.cpu())
            else:
                epoch_loss[l] = [v]

    log_data = {l: torch.tensor(v).mean().item() for l, v in epoch_loss.items()}
    metrics = {k: torch.cat(v).mean().item() for k, v in metrics.items()}
    metrics["distance/mape"] *= 100
    metrics["presence/avg_iou"] *= 100
    if testing:
        metrics["presence/fpr"] *= 100
        metrics["presence/tpr"] *= 100
        metrics["presence/fnr"] *= 100
        metrics["presence/auc"] *= 100
    return {**log_data, **metrics}


def run_epoch_testing(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    training_config,
) -> Dict[str, Number]:

    if not args.quiet:
        dataloader = tqdm.tqdm(dataloader, total=len(dataloader), ncols=100)

    data = {
        'true_ori' : [],
        'true_dist' : [],
        'true_presence' : [],

        'pred_ori' : [],
        'pred_dist' : [],
        'pred_presence' : [],
        'pred_cos' : [],
        'pred_sin' : [],
        'sensor_fov' : [],

    }

    for batch in dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        pred = model(batch["scan_virtual_history"])

        if training_config["train_configs"]["loss_function"]["bidirection"]:
            pred["predicted_direction"] = torch.atan2(pred["sine"], pred["cosine"]) / 2.0

            if training_config["train_configs"]["loss_function"]["verse"]:
                pred["predicted_orientation"] = (
                    pred["verse"].sign().clamp(-1, 0)
                    * torch.pi
                    + pred["predicted_direction"]
                )
            else:
                pred["predicted_orientation"] = pred["predicted_direction"]
        else:
            pred["predicted_orientation"] = torch.atan2(pred["sine"], pred["cosine"])


        # nms = post_processing_utils.CircularTensorNMS(threshold=0.95, min_peak_distance=10)
        # nms_pred = nms.connected_components_nms(pred)

        data['pred_dist'].extend(pred['distance'])
        data['pred_ori'].extend(pred['predicted_orientation'])
        data['pred_presence'].extend(pred['presence'])
        data['pred_sin'].extend(pred['sine'])
        data['pred_cos'].extend(pred['cosine'])
    
        data['true_dist'].extend(batch['humans_distance_sensor'])
        data['true_ori'].extend(batch['humans_relative_bearing_sensor'])
        data['true_presence'].extend(batch['humans_presence_sensor'])
        data['sensor_fov'].extend(batch['camera_fov_mask'])


    for k,v in data.items():
        data[k] = torch.stack(v)

    gt_ori = data['true_ori'][data['true_presence'].bool()]
    gt_ori_mean = circmean(gt_ori, low = -torch.pi, high = torch.pi)
    avg_dummy_error = angular_difference(gt_ori_mean, gt_ori)
    
    gt_dist = data['true_dist'][data['true_presence'].bool()]
    gt_dist_mean = gt_dist.mean()
    avg_dist_error = (gt_dist - gt_dist_mean).abs()
    avg_dist_rel_error = avg_dist_error / gt_dist
    dummy_df = {
        'ori_err' : avg_dummy_error,
        'dist_err' : avg_dist_error,
        'dist_err_rel' : avg_dist_rel_error
    }
    with open("avg_dummy_df.pkl", 'wb') as f:
        pickle.dump(dummy_df, f)

    # exit()
    
    detection_metrics = detection_and_pose_metrics(
        presence_pred_raw=data['pred_presence'],
        presence_true=data['true_presence'],
        dist_pred=data['pred_dist'],
        dist_true=data['true_dist'],
        ori_pred=data['pred_ori'],
        ori_true=data['true_ori'],
    )


    metrics = {
        'presence/ap': detection_metrics['ap'].item(),
        'presence/precisions' : detection_metrics['precisions'].tolist(),
        'presence/recalls' : detection_metrics['recalls'].tolist(),
        'presence/thresholds' : detection_metrics['thresholds'].tolist(),
        'orientation/mae' : detection_metrics['orientation_errors'].mean().item(),
        'orientation/symmetric_mae' : detection_metrics['orientation_errors_sym'].mean().item(),
        'orientation/circular_pearson' : detection_metrics['orientation_pearsons'].mean().item(),
        'distance/mape' : detection_metrics['dist_errors_rel'].mean().item(),
        'distance/mae' : detection_metrics['dist_errors'].mean().item(),
        'position/mse' : detection_metrics['position_errors'].mean().item()
    }
    return {**metrics}, data, detection_metrics['errors_per_thr'], detection_metrics['predictions_per_thr']
