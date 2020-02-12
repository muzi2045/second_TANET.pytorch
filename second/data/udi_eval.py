##! Time: 2020-02-10
##! Author: muzi2045
##! this file used for UDI dataset evaluation, compute the mAp value

import copy
import fire
import argparse
import json
import os
import random
import time
import pickle
import tqdm
from pathlib import Path
import pyquaternion

import numpy as np
from pyquaternion import Quaternion
from typing import Tuple, List, Dict, Any

## nuscenes devkit module
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList

class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)

def config_factory() -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """
    # Check if config exists.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, 'UDI_2020.json')
    assert os.path.exists(cfg_path), \
        'Wrong configuration file'
    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)
    return cfg

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """
    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def load_gt(root_path: str, info_path: str, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """

    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    udi_infos = data["infos"]
    metadata = data["metadata"]
    
    sample_tokens_all = [info['token'] for info in udi_infos]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    sample_tokens = sample_tokens_all
    ## UDI Dataset samples 11428 samples
    all_annotations = EvalBoxes()

    label_root_path = root_path + "/label"
    filenames = os.listdir(label_root_path)
    for filename in tqdm.tqdm(filenames, leave=verbose):
        sample_boxes = []
        index = filename.split("_")[0]
        label_path = label_root_path + "/" + filename
        assert Path(label_path).exists()

        with open(label_path, encoding='utf-8') as f:
            res = f.read()
        result = json.loads(res)
        boxes = result["elem"]
        for box in boxes:
            class_name = box["class"]
            box_loc = box["position"]
            box_size = box["size"]
            yaw = box["yaw"]
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
            sample_boxes.append(
                    box_cls(
                        sample_token=int(index),
                        translation=tuple([box_loc["x"], box_loc["y"], box_loc["z"]]),
                        size=tuple([box_size["width"], box_size["depth"], box_size["height"]]),
                        rotation=tuple(quat),
                        velocity=tuple([0, 0]),
                        num_pts= -1,
                        detection_name=class_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''
                    )
                )
        all_annotations.add_boxes(int(index), sample_boxes)
    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def eval_main(root_path, info_path, version, res_path, eval_set, output_dir):

    cfg = config_factory()
    result_path = res_path
    # Check result file exists.
    assert os.path.exists(
        result_path), 'Error: The result file does not exist!'
    # Make dirs.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Load data.
    print('Initializing nuScenes detection evaluation')
    pred_boxes, meta = load_prediction(result_path, cfg.max_boxes_per_sample, DetectionBox,
                                       verbose=True)
    gt_boxes = load_gt(root_path, info_path, eval_set,
                       DetectionBox, verbose=True)

    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
        "Samples in split doesn't match samples in predictions."

    """
    Performs the actual evaluation.
    :return: A tuple of high-level and the raw metric data.
    """
    start_time = time.time()
    verbose = True
    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    if verbose:
        print('Accumulating metric data...')
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes,
                            class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    if verbose:
        print('Calculating metrics...')
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    # Compute evaluation time.
    metrics.add_runtime(time.time() - start_time)

    # Dump the metric data, meta and metrics to disk.
    if verbose:
        print('Saving metrics to: %s' % output_dir)
    metrics_summary = metrics.serialize()
    metrics_summary['meta'] = meta.copy()
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(output_dir, 'metrics_details.json'), 'w') as f:
        json.dump(metric_data_list.serialize(), f, indent=2)

    # Print high-level metrics.
    print('mAP: %.4f' % (metrics_summary['mean_ap']))
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    for tp_name, tp_val in metrics_summary['tp_errors'].items():
        print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    print('NDS: %.4f' % (metrics_summary['nd_score']))
    print('Eval time: %.1fs' % metrics_summary['eval_time'])

    # Print per-class metrics.
    print()
    print('Per-class results:')
    print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    for class_name in class_aps.keys():
        print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_name, class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err'],
                class_tps[class_name]['vel_err'],
                class_tps[class_name]['attr_err']))

    # return metrics_summary

if __name__ == "__main__":
    fire.Fire(eval_main)
