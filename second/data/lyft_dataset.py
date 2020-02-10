# lyft dataset process module
# modiflied from nuscenes_dataset.py

import json
import pickle
import time
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess

import fire
import numpy as np 

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.dataset import Dataset, register_dataset
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar 
from second.utils.timer import simple_timer

# @register_dataset
class LyftDataset(Dataset):
    NumPointFeatures = 4
    NameMapping = {
        'animal': 'animal',
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'emergency_vehicle': 'emergency_vehicle',
        'motorcycle': 'motorcycle',
        'other_vehicle': 'other_vehicle',
        'pedestrian': 'pedestrain',
        'truck': 'truck'
    }
    DefaultAttribute = {
        "car": "object_action_parked",
        "pedestrain": "object_action_walking",
        "bicycle": "object_action_driving_straight_forward",
        "motorcycle": "object_action_parked",
        "other_vehicle": "object_action_driving_straight_forward",
        "emergency_vehicle": "object_action_driving_straight_forward",
        "truck": "object_action_parked",
        "animal": "",
        "bus": "object_action_driving_straight_forward",
    }

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self._lyft_infos = data["infos"]
        self._lyft_infos = list(
            sorted(self._lyft_infos, key=lambda e: e["timestamp"])
        )
        self._metadata = data["metadata"]
        self._class_names = class_names
        self._prep_func = prep_func
        self._kitti_name_mapping = {
            "car": "car",
            "pedestrain": "pedestrain",
        }
        self.version = self._metadata["version"]
        self.eval_version = "ICLR 2019"
        self._with_velocity = False
    
    def __len__(self):
        return len(self._lyft_infos)

    @property
    def ground_truth_annotations(self):
        pass
    
    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example
    
    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        info = self._lyft_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }
        lidar_path = Path(info['lidar_path'])
        spec_lidar_path = '/home/muzi2045/nvme0/lyft_dataset/v1.01-train/lidar/host-a011_lidar1_1233090652702363606.bin'
        if str(lidar_path) != spec_lidar_path:

            # print("read lidar path:", str(lidar_path))
            points = np.fromfile(
                str(lidar_path), dtype=np.float32).reshape((-1,5))[:, :5]

            points[:, 3] /= 255
            points[:, 4] = 0
            sweep_points_list = [points]
        
            ts = info["timestamp"] / 1e6
            # print("info sweeps:", len(info["sweeps"]))
            for sweep in info["sweeps"]:
                if str(sweep["lidar_path"]) == spec_lidar_path:
                    continue
                points_sweep = np.fromfile(
                    str(sweep["lidar_path"]), dtype=np.float32,
                    count=-1).reshape([-1, 5])[:, :5]
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, 3] /= 255
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    "sweep2lidar_rotation"].T
                points_sweep[:, :3] += sweep["sweep2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)

            points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
            res["lidar"]["points"] = points
        else:
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1,4))[:, :4]
            points[:, 3] /= 255
            res["lidar"]["points"] = points

        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }
        
        if 'gt_boxes' in info:
            # mask = info["num_lidar_pts"] > 0
            # gt_boxes = info["gt_boxes"][mask]
            gt_boxes = info["gt_boxes"]
            # print("gt_boxes:", gt_boxes)
            if self._with_velocity:
                # gt_velocity = info["gt_velocity"][mask]
                gt_velocity = info["gt_velocity"]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                # 'names': info["gt_names"][mask],
                'names': info['gt_names']
            }
        return res

    def evaluation_kitti(self, detections, output_dir):
        pass

    def evaluation_nusc(self, detections, output_dir):
        pass

    def evaluation_lyft(self, detections, output_dir):
        pass


    def evaluation(self, detections, output_dir):
        res_nusc = self.evaluation_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_nusc["result"]["nusc"],
            },
            "detail": {
                "eval.nusc": res_nusc["detail"]["nusc"],
            },
        }
        return res

def _second_det_to_nusc_box(detection):
    from lyft_dataset_sdk.utils.data_classes import Box
    import pyquaternion
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi/2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i,6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def _lidar_nusc_box_to_global(info, boxes, classes, eval_version="ICLR 2019"):
    import pyquaternion
    box_list = []
    for box in boxes:
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # from lyft_dataset_sdk.eval.detection.mAP_eva
        #filter det in ego

        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

def _get_available_scenes(lyft):
    available_scenes = []
    print("total scene num:", len(lyft.scene))
    for scene in lyft.scene:
        scene_token = scene["token"]
        scene_rec = lyft.get('scene', scene_token)
        sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
        sd_rec = lyft.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = lyft.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scenes_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = lyft.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes

def _fill_train_infos(lyft,
                      train_scenes,
                      test = False,
                      max_sweeps=10):
    train_lyft_infos = []
    from pyquaternion import Quaternion
    print("sample number:", len(lyft.sample))
    for sample in prog_bar(lyft.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        sd_rec = lyft.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record = lyft.get('calibrated_sensor', 
                             sd_rec['calibrated_sensor_token'])
        pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = lyft.get_sample_data(cam_front_token)
        assert Path(lidar_path).exists()

        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,
            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],
            "timestamp": sample["timestamp"],
        }

        # print("info:", info)

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = lyft.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":
                sd_rec = lyft.get('sample_data', sd_rec['prev'])
                cs_record = lyft.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = lyft.get_sample_data_path(sd_rec['token'])
                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps
        # print("sweeps:", sweeps)
        if not test:
            annotations = [
                lyft.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [lyft.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in LyftDataset.NameMapping:
                    names[i] = LyftDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations])
        
        
        if sample["scene_token"] in train_scenes:
            # print("sample_scene_token:", sample["scene_token"])
            train_lyft_infos.append(info)
        # else:
            # val_nusc_infos.append(info)
    return train_lyft_infos


def create_lyft_infos(root_path, json_path, max_sweeps=10):
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    # from lyft_dataset_sdk.utils import s
    lyft = LyftDataset(data_path = root_path, json_path=json_path, verbose=True)
    root_path = Path(root_path)
    available_scenes = _get_available_scenes(lyft)
    print(f"available scenes: {len(available_scenes)}")
    available_scene_names = [s["name"] for s in available_scenes]
    available_scene_tokens = [s["token"] for s in available_scenes]
    # print("available_scenes_names:", available_scene_names)
    # train_scenes = list(
    #     filter(lambda x:x in available_scene_names, available_scenes))
    # print(f"train_scenes: {len(train_scenes)}")
    
    # train_scenes = set([
        # available_scenes[available_scene_names.index(s)]["token"]
        # for s in available_scenes
    # ])
    train_scenes = available_scene_tokens
    print(f"train scenes: {len(train_scenes)}")
    train_lyft_infos = _fill_train_infos(
        lyft, train_scenes, False, max_sweeps=max_sweeps)
    metadata = {
        "version": "v1.01-train",
    }

    print(
        f"train sample: {len(train_lyft_infos)}"
    )
    data = {
        "infos": train_lyft_infos,
        "metadata": metadata,
    }
    with open(root_path/ "infos_train.pkl", 'wb') as f:
        pickle.dump(data, f)

def get_box_mean(info_path, class_name="car"):
    with open(info_path, 'rb') as f:
        lyft_infos = pickle.load(f)["infos"]
    
    gt_boxes_list = []
    gt_vels_list = []
    for info in lyft_infos:
        gt_boxes = info["gt_boxes"]
        gt_vels = info["gt_velocity"]
        gt_names = info["gt_names"]
        mask = np.array([s == class_name for s in info["gt_names"]], dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        gt_vels = gt_vels[mask]

        gt_boxes_list.append(gt_boxes.reshape(-1, 7))
        gt_vels_list.append(gt_vels.reshape(-1, 2))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    gt_vels_list = np.concatenate(gt_vels_list, axis=0)
    nan_mask = np.isnan(gt_vels_list[:, 0])
    gt_vels_list = gt_vels_list[~nan_mask]

    return {
        "box3d": gt_boxes_list.mean(0).tolist(),
        "detail": gt_boxes_list
    }

def get_all_box_mean(info_path):
    det_names = set()
    for k, v in LyftDataset.NameMapping.items():
        if v not in det_names:
            det_names.add(v)
    det_names = sorted(list(det_names))
    res = {}
    details = {}
    for k in det_names:
        result = get_box_mean(info_path, k)
        details[k] = result["detail"]
        res[k] = result["box3d"]
    print(json.dumps(res, indent=2))
    return details


if __name__ == "__main__":
    fire.Fire()






