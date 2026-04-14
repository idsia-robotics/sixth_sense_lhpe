from typing import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from numpy import mean
import numpy as np
import torch
from lidar_human_pose_estimation.utils import post_processing_utils
from lidar_human_pose_estimation.utils.geom_utils import detection_matching
from lidar_human_pose_estimation.utils.matching_utils import match_detections
from munkres import Munkres

def angular_difference(x, y):
    """
    Circular difference between each angle in a and b.
    Returns a new tensor with the same shape as `a` and `b` where
    each element i is the angular difference between a_i and b_i.
    """

    def custom_modulo(a, b):
        return a - torch.floor(a / b) * b

    return torch.abs(custom_modulo(x - y + torch.pi, torch.pi * 2) - torch.pi)


def orientation_absolute_error(
    pred: torch.Tensor,
    true: torch.Tensor,
    human_presence_mask: torch.BoolTensor,
    symmetric=False,
    reduce = False
):
    if human_presence_mask.any():
        if symmetric:
            ang1 = angular_difference(
                pred[human_presence_mask], true[human_presence_mask]
            )
            ang2 = angular_difference(
                pred[human_presence_mask], true[human_presence_mask] + torch.pi
            )
            if not reduce:
                return torch.minimum(ang1, ang2)
            else:
                return torch.minimum(ang1, ang2).mean()
        else:
            if not reduce:
                return angular_difference(
                    pred[human_presence_mask], true[human_presence_mask]
                )
            else:
                return angular_difference(
                    pred[human_presence_mask], true[human_presence_mask]
                ).mean()
    else:
        return torch.tensor([], device=true.device)


def distance_ape(
    pred: torch.Tensor, true: torch.Tensor, human_presence_mask: torch.BoolTensor, reduce = False
):
    """
    Returns the Absolute Percentual Error (MAPE) of distance estimations.
    The returned value is a percentage expressed in the [0,1] value range.
    """
    true = true[human_presence_mask]
    if true.numel() > 0:
        pred = pred[human_presence_mask]
        true[true == 0] += 1e-12
        if not reduce:
            return pred.sub(true).abs().div(true)
        else:
            return pred.sub(true).abs().div(true).mean()
    else:
        return torch.tensor([], device=true.device)


def distance_mae(
    pred: torch.Tensor, true: torch.Tensor, human_presence_mask: torch.BoolTensor, reduce = False
):
    true = true[human_presence_mask]
    if true.numel() > 0:
        pred = pred[human_presence_mask]
        true[true == 0] += 1e-12
        if not reduce:
            return pred.sub(true).abs()
        else:
            return pred.sub(true).abs().mean()
    else:
        return torch.tensor([], device=true.device)



def iou(a, b):
    union = (a + b).sum(-1)
    result = a.mul(b).sum(-1).div(union.add(1e-10))
    return result


def presence_average_iou(presence_pred_raw, presence_gt, fov_mask, reduce=False):
    thrs = torch.arange(0.5, 0.901, 0.1).to(presence_pred_raw.device)
    thrs = thrs[:, None, None]
    masked_preds = (presence_pred_raw > thrs).float()
    presence_gt = presence_gt[None, ...]
    fov_mask = fov_mask[None, ...]
    ious = iou(masked_preds.mul(fov_mask), presence_gt.mul(fov_mask)).mean((-1))
    # IoUs = []
    # for t in thrs:
        # presence_pred_bin = presence_pred_raw > t
        # IoUs.append(iou(presence_pred_bin * fov_mask, presence_gt * fov_mask).mean())

    if not reduce:
        return ious
    else:
        return ious.mean()

def __circular_pearson_torch(x, y):
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor([], device=x.device)
    else:
        mean_x = x.mean()
        mean_y = y.mean()
        x_diff = torch.sin(x - mean_x)
        y_diff = torch.sin(y - mean_y)
        if (y_diff == 0).all():
            return torch.tensor([], device=x.device)
        res = x_diff.mul(y_diff).sum() / torch.sqrt(((x_diff**2).sum() * (y_diff**2).sum()))[None, ...]
        return res
    
def circular_pearson(x, y):
    if isinstance(x, torch.Tensor) or isinstance(x, torch.Tensor):
        return __circular_pearson_torch(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_diff = np.sin(x - mean_x)
    y_diff = np.sin(y - mean_y)
    return (x_diff * y_diff).sum() / np.sqrt(((x_diff**2).sum() * (y_diff**2).sum()))

def __pres_dist_to_vec(pres, dist):
    batch_idxs, in_batch_idxs = torch.where(pres > 0)
    result = [[] for _ in range(pres.shape[0])]
    if len(batch_idxs) == 0:
        return result

    presence_angles = torch.deg2rad(in_batch_idxs.flatten())
    dist = dist[pres > 0].flatten()
    vectors = torch.stack([torch.cos(presence_angles) * dist, torch.sin(presence_angles) * dist], dim = 1)
    for id,v in zip(batch_idxs, vectors):
        result[id].append(v)

    for i in range(len(result)):
        if len(result[i]) > 0:
            result[i] = torch.stack(result[i])
        else:
            result[i] = torch.tensor([])
    return result

def human_detection_precision(presence_pred, presence_true, dist_pred, dist_true, ori_pred, ori_true, distance_threshold_m = 1.5):


    true_human_vectors = __pres_dist_to_vec(presence_true, dist_true) # BxL_kx2
    pred_human_vectors = __pres_dist_to_vec(presence_pred, dist_pred) # BxM_kx2
    pred_M = max([len(x) for x in pred_human_vectors])
    true_M = max([len(x) for x in true_human_vectors])

    true_human_vectors_tens = torch.full((len(true_human_vectors), true_M, 2), torch.nan)
    pred_human_vectors_tens = torch.full((len(pred_human_vectors), pred_M, 2), torch.nan)

    for i in range(len(true_human_vectors)):
        if len(true_human_vectors[i]) > 0:
            true_human_vectors_tens[i, :len(true_human_vectors[i]), ...] = true_human_vectors[i]
        if len(pred_human_vectors[i]) > 0:
            pred_human_vectors_tens[i, :len(pred_human_vectors[i]), ...] = pred_human_vectors[i]

    tps = 0
    fps = 0
    fns = 0
    distance_errors = []
    distance_errors_rel = []
    tp_orientation_preds = []
    tp_orientation_trues = []
    tp_matching_costs = []
    tp_dist_preds = []
    tp_dist_trues = []

    matched_detections = match_detections(detections_gt=true_human_vectors_tens,
                                          detections_pred=pred_human_vectors_tens,
                                           max_cost = distance_threshold_m)


    for b_id in range(presence_pred.shape[0]): # Iterate batches

        matched_gts = matched_detections[b_id]['gt_TP_ids']
        matched_preds = matched_detections[b_id]['pred_TP_ids']
        
        if len(matched_gts) == 0:
            fps += len(pred_human_vectors[b_id])
            continue
        elif len(matched_preds) == 0:
            fns += len(true_human_vectors[b_id])
            continue
        
        tps += len(matched_detections[b_id]['pred_TP_ids'])
        fps += len(matched_detections[b_id]['pred_FP_ids'])
        fns += len(matched_detections[b_id]['gt_FN_ids'])

        tp_dist_pred = dist_pred[b_id][presence_pred[b_id] > 0][matched_preds]
        tp_dist_true = dist_true[b_id][presence_true[b_id] > 0][matched_gts]

        distance_errors.extend((tp_dist_pred - tp_dist_true).abs())
        distance_errors_rel.extend(((tp_dist_pred - tp_dist_true) / tp_dist_true).abs())
        
        tp_ori_pred = ori_pred[b_id][presence_pred[b_id] > 0][matched_preds]
        tp_ori_true = ori_true[b_id][presence_true[b_id] > 0][matched_gts]

        tp_orientation_preds.extend(tp_ori_pred)
        tp_orientation_trues.extend(tp_ori_true)

        tp_dist_trues.extend(tp_dist_true)
        tp_dist_preds.extend(tp_dist_pred)
        
        tp_matching_costs.extend([c for p, g, c in matched_detections[b_id]['matched_pairs']])



    if not len(tp_orientation_preds) == 0:
        tp_orientation_preds = torch.stack(tp_orientation_preds)
        tp_orientation_trues = torch.stack(tp_orientation_trues)
        tp_matching_costs = torch.stack(tp_matching_costs)
        tp_dist_trues = torch.stack(tp_dist_trues)
        tp_dist_preds = torch.stack(tp_dist_preds)
    else:
        tp_orientation_preds = torch.tensor([1., 0.])
        tp_orientation_trues = torch.tensor([1., 0.])
        tp_matching_costs = torch.tensor([0.])
        tp_dist_trues = torch.tensor([0.])
        tp_dist_preds = torch.tensor([0.])
    
    return {
        'tps' : tps,
        'fps' : fps,
        'fns' : fns,
        'tp_dist_errors' : torch.tensor(distance_errors),
        'tp_dist_errors_rel' : torch.tensor(distance_errors_rel),
        'tp_ori_errors' : orientation_absolute_error(
            pred=tp_orientation_preds,
            true=tp_orientation_trues,
            human_presence_mask=torch.ones_like(tp_orientation_trues, dtype = torch.bool)

        ),
        'tp_ori_errors_sym' : orientation_absolute_error(
            pred=tp_orientation_preds,
            true=tp_orientation_trues,
            human_presence_mask=torch.ones_like(tp_orientation_trues, dtype = torch.bool),
            symmetric=True

        ),
        'tp_circular_pearson' : circular_pearson(
            tp_orientation_preds, tp_orientation_trues
        ),
        'tp_position_error' : tp_matching_costs,
        'tp_ori_trues' : tp_orientation_trues,
        'tp_ori_preds' : tp_orientation_preds,
        'tp_dist_trues' : tp_dist_trues,
        'tp_dist_preds' : tp_dist_preds
    }


def detection_and_pose_metrics(presence_pred_raw, presence_true, dist_pred, dist_true, ori_pred, ori_true):
    """
    Computes detection metrics over multiple NMS thresholds.
    For each threshold, the regressed pose's accuracy is given as an orientation and distance error.
    """
    thrs = np.concatenate([np.linspace(.1, .9, 9), np.linspace(.9, .95, 10), np.linspace(.95, .99, 10)[1:]], )
    # sp = np.linspace(.1, 1, 10)
    # sp[-1] = .99

    # thrs = np.quantile(presence_pred_raw.flatten(), q = sp)
    
    precisions = []
    recalls = []
    dist_errors = OrderedDict()
    dist_errors_rel = OrderedDict()
    orientation_errors = OrderedDict()
    orientation_errors_sym = OrderedDict()
    orientation_pearsons = OrderedDict()
    position_errors = OrderedDict()
    
    ori_preds = OrderedDict()
    ori_trues = OrderedDict()
    dist_preds = OrderedDict()
    dist_trues = OrderedDict()

    for thr in thrs:

        nms = post_processing_utils.CircularTensorNMS(threshold=thr)
        pred_dict = {
            'presence' : presence_pred_raw,
            'distance' : dist_pred,
            'orientation' : ori_pred
            # 'sine' : torch.ones_like(presence_pred_raw),
            # 'cosine' : torch.ones_like(presence_pred_raw),
        }

        pred_dict = nms.iterative_peak_nms(pred_dict)
        presence_pred = pred_dict['presence']
        m = human_detection_precision(
            presence_pred=presence_pred,
            presence_true=presence_true,
            dist_pred=pred_dict['distance'],
            dist_true=dist_true,
            ori_true=ori_true,
            ori_pred=pred_dict['orientation'],
            distance_threshold_m=0.5
        )
        print(f"TPs @{thr} = {m['tps']}")
        precision = m['tps'] / (m['tps'] + m['fps'] + 1e-10)
        recall = m['tps'] / (m['tps'] + m['fns'] + 1e-10)
        precisions.append(precision)
        recalls.append(recall)
        if m['tps'] > 0:
            dist_errors[thr] = m['tp_dist_errors']
            dist_errors_rel[thr] = m['tp_dist_errors_rel']
            orientation_errors[thr] = m['tp_ori_errors']
            orientation_errors_sym[thr] = m['tp_ori_errors_sym']
            orientation_pearsons[thr] = m['tp_circular_pearson']
            position_errors[thr] = m['tp_position_error']
            ori_preds[thr] = m['tp_ori_preds']
            ori_trues[thr] = m['tp_ori_trues']
            dist_preds[thr] = m['tp_dist_preds']
            dist_trues[thr] = m['tp_dist_trues']
    
    
    
    errors_per_thr = {
        'dist' : dist_errors,
        'dist_rel' : dist_errors_rel,
        'position' : position_errors,
        'orientation' : orientation_errors,
        'orientation_sym' : orientation_errors_sym,
        'orientation_pearson' : orientation_pearsons
    },

    predictions_per_thr = {
        'dist_pred' : dist_preds,
        'dist_true' : dist_trues,
        'ori_pred'  : ori_preds,
        'ori_true' : ori_trues
    }

    precisions = np.flip(np.array(precisions))
    recalls = np.flip(np.array(recalls))
    ap = (np.diff(recalls) * np.flip(precisions[1:, ])).sum()
    thrs = np.flip(thrs)
    dist_errors = np.flip(np.array([v.mean() for v in dist_errors.values()]))
    dist_errors_rel = np.flip(np.array([v.mean() for v in dist_errors_rel.values()]))
    position_errors = np.flip(np.array([v.mean() for v in position_errors.values()]))
    orientation_errors = np.flip(np.array([v.mean() for v in orientation_errors.values()]))
    orientation_errors_sym = np.flip(np.array([v.mean() for v in orientation_errors_sym.values()]))
    orientation_pearsons = np.flip(np.array([v.mean() for v in orientation_pearsons.values()]))
    
    return {
        'precisions' : precisions,
        'recalls' : recalls,
        'ap' : ap,
        'thresholds' : thrs,
        'dist_errors' : dist_errors,
        'dist_errors_rel' : dist_errors_rel,
        'orientation_errors' : orientation_errors,
        'orientation_errors_sym' : orientation_errors_sym,
        'orientation_pearsons' : orientation_pearsons,
        'position_errors' : position_errors,
        'errors_per_thr' : errors_per_thr,
        'predictions_per_thr' : predictions_per_thr
    }
