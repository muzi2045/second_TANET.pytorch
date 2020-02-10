# Simple SECOND inference code
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import ros_numpy
import numpy as np
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from pyquaternion import Quaternion

import sys
import os
import pickle
import shutil
import time
import math
import torch

from pathlib import Path
from google.protobuf import text_format
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
from second.core import box_np_ops

_NEXT_AXIS = [1, 2, 0, 1]


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[1,0,0], angle=-yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.55, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(1, 0.15, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(2, 0.15, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(3, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(4, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(5, 0.22, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(6, 0.2, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(7, 0.3, label_preds_, scores_)
    truck_indices =                get_annotations_indices(8, 0.4 , label_preds_, scores_)
    barrier_indices =              get_annotations_indices(9, 0.4 , label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations

def remove_low_score_udi(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =          get_annotations_indices(0, 0.58, label_preds_, scores_)
    pedestrian_indices =   get_annotations_indices(1, 0.15, label_preds_, scores_)
    cyclist_indices =      get_annotations_indices(2, 0.15, label_preds_, scores_)
    truck_indices =        get_annotations_indices(3, 0.3, label_preds_, scores_)
    forklift_indices =     get_annotations_indices(4, 0.3, label_preds_, scores_)
    golfcar_indices =      get_annotations_indices(5, 0.3, label_preds_, scores_)
    motorcyclist_indices = get_annotations_indices(6, 0.2, label_preds_, scores_)
    bicycle_indices =      get_annotations_indices(7, 0.2, label_preds_, scores_)
    motorbike_indices =    get_annotations_indices(8, 0.2 , label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrian_indices + 
                            cyclist_indices +
                            truck_indices +
                            forklift_indices +
                            golfcar_indices +
                            motorcyclist_indices +
                            bicycle_indices +
                            motorbike_indices  
                            ])

    return img_filtered_annotations

def remove_low_score_lyft(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.5, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(1, 0.15, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(2, 0.15, label_preds_, scores_)
    animal_indices =               get_annotations_indices(3, 0.1, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(4, 0.1, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(5, 0.0, label_preds_, scores_)
    other_vehicle_indices =        get_annotations_indices(6, 0.2, label_preds_, scores_)
    emergency_vehicle_indices =    get_annotations_indices(7, 0.2, label_preds_, scores_)
    truck_indices =                get_annotations_indices(8, 0.2, label_preds_, scores_)
    # print("truck indices:", truck_indices)
    # barrier_indices =              get_annotations_indices(9, 0.4 , label_preds_, scores_)
    
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            animal_indices +
                            motorcycle_indices +
                            other_vehicle_indices +
                            emergency_vehicle_indices+
                            truck_indices
                            ])

    return img_filtered_annotations

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.input_cfg = None
        self.model_cfg = None
        self.device = None
        self.net = None
        self.target_assigner = None
        self.voxel_generator = None
        self.anchors = None  
        self.inputs = None
        
    def initialize(self):
        self.read_config()
        self.build_network()
        self.generate_anchors()
        
    def read_config(self):
        config_path = self.config_path
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        self.input_cfg = config.eval_input_reader
        self.model_cfg = config.model.second
        config_tool.change_detection_range(self.model_cfg, [-30, -20, 30, 20])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_network(self):
        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(self.model_path))
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

    def generate_anchors(self):
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]
        anchors = self.target_assigner.generate_anchors(feature_map_size)["anchors"]
        anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        anchors = anchors.view(1, -1, 7)
        self.anchors = anchors

    def run(self, points):
        #t_t = time.time()      
        num_features = 4        
        self.points = points.reshape([-1, num_features])
        self.voxel_generator._to_sparse = True
        print("  input points shape:", self.points.shape)

        t = time.time()
        voxel_output = self.voxel_generator.generate(self.points, max_voxels=10000)
        print("  generate voxels cost time:", time.time() - t)
        
        voxels = voxel_output["voxels"]
        coords = voxel_output["coordinates"]
        num_points = voxel_output["num_points_per_voxel"]
        print("  voxel shape:", voxels.shape)
        print("  coords shape:", coords.shape)
        
        t = time.time()
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        torch.cuda.synchronize()
        print("  voxels convert to tensor time: ", time.time() - t)

        torch.cuda.synchronize()
        t = time.time()
        self.inputs = {
            "anchors": self.anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords,
        }
        pred = self.net(self.inputs)[0]
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        # pred = remove_low_score_lyft(pred, 0.45)
        # pred = remove_low_score_nu(pred, 0.45)
        pred = remove_low_score_udi(pred, 0.45)
   
        #t = time.time()
        boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        # point_indices = box_np_ops.points_in_rbbox(self.points, boxes_lidar, origin=(0.5, 0.5, 0.5))

        # print("point_indices:", point_indices.shape)

        scores = pred["scores"].detach().cpu().numpy()
        types = pred["label_preds"].detach().cpu().numpy()
        #print("  return result time:", time.time() - t)
        
        #print("  total_run_time: ", time.time() - t_t)
        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    print("cluster points shape:", points_sum.shape)
    # if len(points_sum.shape) == 3:
    msg.height = 1
    msg.width = points_sum.shape[1] * points_sum.shape[0]
        # msg.width = points_sum.shape[1]
    # else:
    #     msg.height = 1
    #     msg.width = len(points_sum)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 4
    msg.row_step = points_sum.shape[1]
    msg.is_dense = int(np.isfinite(points_sum).all())
    # msg.is_dense = 0
    print("msg.is_dense:", msg.is_dense)
    # msg.data = np.asarray(points_sum, np.float32).tostring()
    msg.data = points_sum.astype(np.float32).tobytes()
    # print("convert msg.data:", msg.data)
    return msg


def rslidar_callback(msg):
    t_t = time.time()
    
    arr_bbox = BoundingBoxArray()
    
    #t = time.time()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    #print("prepare cloud time: ", time.time() - t)
       
    #t = time.time()
    scores, dt_box_lidar, types = proc_1.run(np_p)
    #print("network forward time: ", time.time() - t)
    # filter_points_sum = []
    #t = time.time()
    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            
            q = yaw2quaternion(float(dt_box_lidar[i][6]))
            bbox.pose.orientation.x = q[0]
            bbox.pose.orientation.y = q[1]
            bbox.pose.orientation.z = q[2]
            bbox.pose.orientation.w = q[3]
           

            bbox.pose.position.x = float(dt_box_lidar[i][0])
            bbox.pose.position.y = float(dt_box_lidar[i][1])
            bbox.pose.position.z = float(dt_box_lidar[i][2])
            bbox.dimensions.x = float(dt_box_lidar[i][3])
            bbox.dimensions.y = float(dt_box_lidar[i][4])
            bbox.dimensions.z = float(dt_box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = types[i]
            arr_bbox.boxes.append(bbox)

            # filter_points = np_p[point_indices[:,i]]
            # filter_points_sum.append(filter_points)
    #print("publish time cost: ", time.time() - t)
    # filter_points_sum = np.concatenate(filter_points_sum, axis=0)
    # filter_points_sum = filter_points_sum[:, :3]

    # print("output of concatenate:", filter_points_sum)
    # filter_points_sum = np.arange(24).reshape(8,3)
    # cluster_cloud = xyz_array_to_pointcloud2(filter_points_sum, stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
    # pub_segments.publish(cluster_cloud)

    print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp = rospy.Time.now()
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)

if __name__ == '__main__':

    # q = quaternion_from_euler(0,0,3.14)
    # print("q:"+ str(q[0]) + "-" + str(q[1]) + "-" + str(q[2]) + "-" + str(q[3]))
    global proc
    
    # all 10 classes detection model( 2019-4-29)
    # config_path = '/home/aisimba/Documents/second.pytorch-v1.6/second/trained_model/all_pp_lowa_nu/pipeline.config'
    # model_path = '/home/aisimba/Documents/second.pytorch-v1.6/second/trained_model/all_pp_lowa_nu/voxelnet-586500.tckpt'

    # all 10 classes detection model( 2019-5-05)
    # config_path = '/home/muzi2045/Documents/second.pytorch/second/trained_model/nu_pp_2019_11_18/pipeline.config'
    # model_path = '/home/muzi2045/Documents/second.pytorch/second/trained_model/nu_pp_2019_11_18/voxelnet-562680.tckpt'

    # pointpillars_with TA module detection model( can't work) (2020-01-20)
    # config_path = '/home/muzi2045/Documents/second_TANET.pytorch/second/trained_model/tanet_2020_01_20/pipeline.config'
    # model_path = '/home/muzi2045/Documents/second_TANET.pytorch/second/trained_model/tanet_2020_01_20/voxelnet-562680.tckpt'

    ## pointpillars UDI dataset detection model 2020-02-08
    config_path = '/home/muzi2045/Documents/second_TANET.pytorch/second/trained_model/pp_udi_2020_02_07/pipeline.config'
    model_path = '/home/muzi2045/Documents/second_TANET.pytorch/second/trained_model/pp_udi_2020_02_07/voxelnet-562680.tckpt'

    proc_1 = Processor_ROS(config_path, model_path)
    
    proc_1.initialize()
    
    rospy.init_node('second_ros_node')
    # sub_ = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, rslidar_callback, queue_size=1, buff_size = 2**24)
    sub_ = rospy.Subscriber("/top/rslidar_points", PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    # sub_ = rospy.Subscriber("/merged_cloud", PointCloud2, rslidar_callback, queue_size=1, buff_size = 2**24)
    #sub_ = rospy.Subscriber("/roi_pclouds", PointCloud2, rslidar_callback, queue_size=1, buff_size = 2**24)
    
    pub_arr_bbox = rospy.Publisher("second_arr_bbox", BoundingBoxArray, queue_size=1)
    # pub_segments = rospy.Publisher("second_clusters", PointCloud2, queue_size=1)
    # pub_cluster = rospy.Publisher("rslidar_points_modified", PointCloud2, queue_size=1, buff_size = 2**24)

    print("[+] second_ros_node has started!")    
    rospy.spin()
