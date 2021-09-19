from typing import Tuple, List, Dict
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import Interface
from DataDecoder import RadarPointCloud,HostInfo
import os.path as osp
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from VisualPlot import VisualPlot
from pyquaternion import Quaternion
import pandas as pd 
import configparser
from functools import reduce



class DataLoader:
    """
    DataLoader Class to 
    """
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('D:/Tech_Resource/Paper_Resource/Perception_R_C_Fusion_with_ShizhanWang_Project/code/config.ini')
        self.version = self.config['Default']['Version']
        self.dataroot = self.config['Default']['Dataroot']
        self.versbose = self.config.getboolean('Default','Versbose')
    
    #find closest timestamp that is smaller than the base timestamp 
    def findClosest(self,
                    base_timestamp:int,
                    data_array:dict)->int:  
        
        ind =  min(range(len(data_array)), key = lambda i: abs(list(data_array.keys())[i]-base_timestamp))
        
        if list(data_array.keys())[ind] > base_timestamp and ind !=0:
            ind = ind - 1
        else:
            pass
        return ind
    
    #Make interface data's ndarray read-only
    def arrayImmu(self):
        self.front_radar_data.pc.flags.writeable = False
        self.front_radar_data.rotation.flags.writeable = False
        self.front_radar_data.translation.flags.writeable = False
        self.FL_radar_data.pc.flags.writeable = False
        self.FL_radar_data.rotation.flags.writeable = False
        self.FL_radar_data.translation.flags.writeable = False
        self.FR_radar_data.pc.flags.writeable = False
        self.FR_radar_data.rotation.flags.writeable = False
        self.FR_radar_data.translation.flags.writeable = False
        self.front_cam_data.rotation.flags.writeable = False
        self.front_cam_data.translation.flags.writeable = False
        self.front_cam_data.intrinsic.flags.writeable = False
    
    #Logging
    def log(self, 
            time:int):

        self.out_path_for_logs = self.config['Log']['OutPath4Log']
        self.out_path_for_figures = self.config['Log']['OutPath4Fig']
        log_radar_pc=self.config.getboolean('Log', 'Radar_pc')
        log_bbox = self.config.getboolean('Log', 'Bbox')
        log_host = self.config.getboolean('Log', 'Host')
        shouldLog = self.config.getboolean('Log', 'ShouldLog')
        # Logging switch
        if shouldLog:
            scene_name = self.current_scene['name']
            if log_radar_pc:
                pd_data = pd.DataFrame(self.front_radar_data.pc, index=['x','y','z','dyn_prop','id','rcs','vx','vy',
                                                                        'vx_comp','vy_comp','is_quality_valid',
                                                                        'ambig_state','x_rms','y_rms','invalid_state',
                                                                        'pdh0','vx_rms','vy_rms'])
                pd_data.to_csv(self.out_path_for_logs+'/'+ scene_name +'_f_radar_%d.csv' % time, index_label = 'variable/sequence')
                pd_data = pd.DataFrame(self.FL_radar_data.pc, index=['x','y','z','dyn_prop','id','rcs','vx','vy',
                                                                        'vx_comp','vy_comp','is_quality_valid',
                                                                        'ambig_state','x_rms','y_rms','invalid_state',
                                                                        'pdh0','vx_rms','vy_rms'])
                pd_data.to_csv(self.out_path_for_logs+'/'+ scene_name +'_FL_radar_%d.csv' % time, index_label = 'variable/sequence')
                pd_data = pd.DataFrame(self.FR_radar_data.pc, index=['x','y','z','dyn_prop','id','rcs','vx','vy',
                                                                        'vx_comp','vy_comp','is_quality_valid',
                                                                        'ambig_state','x_rms','y_rms','invalid_state',
                                                                        'pdh0','vx_rms','vy_rms'])
                pd_data.to_csv(self.out_path_for_logs+'/'+ scene_name +'_FR_radar_%d.csv' % time, index_label = 'variable/sequence')
            if log_host:
                pd_data = pd.DataFrame(self.host_array, columns=['time','x','y','z','vx','vy','vz','ax','ay','az',
                                                                 'yawrate','rotation_vector'])
                pd_data.to_csv(self.out_path_for_logs+'/'+ scene_name + '_host.csv', index_label = 'sequence/variable')
            if log_bbox:
                pd_data = pd.DataFrame(columns=['id','x','y','z','vx','vy','vz','orien','w','l','h','category'])
                for i in range(len(self.gt.bboxes)):
                    pd_data.loc[i] = self.gt.bboxes[i].to_list()
                pd_data.to_csv(self.out_path_for_logs+'/'+ scene_name + '_groundTruth_%d.csv' % time, index_label = 'sequence/variable')
    
    #assigna ground truth data structure
    def ground_truth_assign(self,
                            token:str):
        id_inc = 0
        self.gt = Interface.GroundTruth(bboxes=[])
        for box in self.nusc.get_boxes(token):
            bbox = Interface.Bbox(id=id_inc,x=box.center[0],y=box.center[1], z=box.center[2],vx=box.velocity[0],vy=box.velocity[1],
                                  vz=box.velocity[0],w=box.wlh[0],l=box.wlh[1],h=box.wlh[2],orien=box.orientation.degrees,
                                  categr=box.name)
            self.gt.bboxes.append(bbox)
            id_inc+=1

    
    def run(self):
        #NuScenes initialization
        self.nusc = NuScenes(self.version, self.dataroot, self.versbose)
        #NuScenes CAN initialization
        nusc_can = NuScenesCanBus(self.dataroot)
        #First scene
        self.current_scene = self.nusc.scene[0]
        #Retrieve host array with sampling rate of 100 Hz;
        #host_time_dict: Dictionary of timestamp to pose matrix row index (data sequence)
        self.host_array, host_time_dict = HostInfo.fromScene(nusc_can, self.current_scene['name'],'pose')
        #Find the first sample token
        first_sample_token = self.current_scene['first_sample_token']
        first_sample = self.nusc.get('sample', first_sample_token)

        #Dictionary of timestampsto front/FL/FR radar token
        front_radar_time_dict = dict()
        FL_radar_time_dict = dict()
        FR_radar_time_dict = dict()
        front_radar_rec = self.nusc.get('sample_data', first_sample['data']['RADAR_FRONT'])
        FL_radar_rec = self.nusc.get('sample_data', first_sample['data']['RADAR_FRONT_LEFT'])
        FR_radar_rec = self.nusc.get('sample_data', first_sample['data']['RADAR_FRONT_RIGHT'])
        while first_sample:
            front_radar_time_dict[front_radar_rec['timestamp']] = front_radar_rec['token']
            FL_radar_time_dict[FL_radar_rec['timestamp']] = FL_radar_rec['token']
            FR_radar_time_dict[FR_radar_rec['timestamp']] = FR_radar_rec['token']
            if(front_radar_rec['next'] == '' or FL_radar_rec['next'] =='' or FR_radar_rec['next'] ==''):
                break
            else:
                front_radar_rec = self.nusc.get('sample_data', front_radar_rec['next'])
                FL_radar_rec = self.nusc.get('sample_data', FL_radar_rec['next'])
                FR_radar_rec = self.nusc.get('sample_data', FR_radar_rec['next'])
        list_of_radar_time_token_pair = [front_radar_time_dict, FL_radar_time_dict, FR_radar_time_dict]        
        #synchronization based on front camera time stamp
        current_sd_rec = self.nusc.get('sample_data', first_sample['data']['CAM_FRONT']) 
        self.loop_counter = 0 #sequence counter  
        prev_timestamp = 0 # previous loop's timestamp
        while first_sample:
            front_cam_time = current_sd_rec['timestamp'] #front camera time stamp
            
            #find front host time stamp which is the closest to the selected camera time stamp, front_cam_time
            host_ind = self.findClosest(front_cam_time, host_time_dict)
            host_time = list(host_time_dict.keys())[host_ind]
            host_rec = self.host_array[host_ind,:]
            ##host data structure
            t_x = host_rec[1]
            t_y = host_rec[2]
            t_z = host_rec[3]
            t_vx = host_rec[4]
            t_vy = host_rec[5]
            t_vz = host_rec[6]
            t_time_stamp = host_rec[0] # t_time_stamp is same as host_time now
            #host data synchronization by linear interpolation
            #TODO: Interpolation may be applied to all attributes than only to x, y and z? e.g. also interpolate the host vx, vy, vz
            if self.config.getboolean('Preprocessor','ShouldInterpl'):
                delta_time = (front_cam_time - t_time_stamp)*1e-6
                if self.loop_counter>1: 
                    assert delta_time>=0, 'delta time cannot be negative'
                t_x+=(t_vx*delta_time)
                t_y+=(t_vy*delta_time)
                t_z+=(t_vz*delta_time)
                t_time_stamp = front_cam_time
            # get the host data by filling into correct data
            self.host_data = Interface.HostData(time_stamp=t_time_stamp, x=t_x, y=t_y, z=t_z,
                                           vx=t_vx, vy=t_vy, vz=t_vz, ax=host_rec[7], ay= host_rec[8],
                                           az=host_rec[9], yaw_rate=host_rec[10], rotation=host_rec[11])
            
            # Acquire front camera intrinsic and extrinsic calibration parameters at time front_cam_time
            cam_cs_record = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            # Get ego vehicle pose in the world coordinate frame at time front_cam_time
            ref_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            # Homogeneous transform from ego car coordinate frame to front camera coordinate frame at time front_cam_time
            front_camera_from_ego_at_front_cam_time = transform_matrix(cam_cs_record['translation'], Quaternion(cam_cs_record['rotation']), 
                                                inverse=True)
            # Homogeneous transformation matrix from world coordinate to current ego car coordinate frame at time front_cam_time
            ego_from_world_at_front_cam_time = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=True)
            
            data_path, boxes, _ = self.nusc.get_sample_data(current_sd_rec['token'], box_vis_level=BoxVisibility.ANY)
            data = Image.open(data_path)
            ##front camera data structure
            self.front_cam_data = Interface.CameraData(front_cam_time,0,data,np.array(cam_cs_record['camera_intrinsic']),
                                                  np.array(cam_cs_record['rotation']), np.array(cam_cs_record['translation']),
                                                   current_sd_rec['token'])
            
            radar_interface_names = ['front_radar_data', 'FL_radar_data', 'FR_radar_data'] # Config the radar interface, just use 'front_radar_data' if we do not want other radar data
            radar_data = dict()
            for i, sensor_name in enumerate(radar_interface_names): # for front radar, front left radar and front right radar
                #find radar time stamp which is the closest to the selected camera time stamp, front_cam_time
                radar_time = list(list_of_radar_time_token_pair[i].keys())[self.findClosest(front_cam_time, list_of_radar_time_token_pair[i])]     
                #find radar token based on time stamp      
                radar_token = list_of_radar_time_token_pair[i][radar_time]
                #Acquire radar record
                radar_rec = self.nusc.get('sample_data', radar_token) 
                
                if self.config.getboolean('Preprocessor', 'AccumulateMultipleRadarScans'): # Accumulate radar detection points from multiple time scans together here
                    number_of_radar_time_scans_for_accumulation = self.config.getint('Preprocessor', 'NumberOfRadarTimeScansForAccumulation')

                    # Accumulate current and previous number_of_radar_time_scans_for_accumulation sweeps(time scans).
                    for time_scan in range(number_of_radar_time_scans_for_accumulation):
                        # Retrieve radar point cloud data from current radar data
                        file_name = osp.join(self.nusc.dataroot, radar_rec['filename'])
                        RadarPointCloud.disable_filters()  #disable filters(so extract all the detection points from RadarPointCloud)
                        current_pc = RadarPointCloud.from_file(file_name)

                        # Get past ego pose in the world coordinate frame at current radar time
                        current_pose_rec = self.nusc.get('ego_pose', radar_rec['ego_pose_token'])
                        # Homogeneous transformation matrix from current ego car coordinate frame to world coordinate at current radar time
                        world_from_ego_at_current_radar_time = transform_matrix(current_pose_rec['translation'],
                                                        Quaternion(current_pose_rec['rotation']), inverse=False)

                        # Get radar extrinsic calibration parameters
                        radar_cal = self.nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])
                        # Homogeneous transform from current radar coordinate frame to ego car coordinate frame at current radar time
                        ego_from_radar_at_current_radar_time = transform_matrix(radar_cal['translation'], Quaternion(radar_cal['rotation']),
                                                            inverse=False)

                        if self.config.getboolean('Preprocessor', 'ShouldInterpl'): # Interpolate to synchronize multiple time scans data in both space and time 
                            delta_time = (front_cam_time - radar_time)*1e-6 # Calculate time difference
                            # Multiply 3 transformation matrices into one--trans_matrix_ego_from_radar_in_different_time
                            trans_matrix_ego_from_radar_in_different_time = reduce(np.dot, [ego_from_world_at_front_cam_time, 
                                                                                world_from_ego_at_current_radar_time, ego_from_radar_at_current_radar_time])
                            # Calculate position information x, y, for the radar detection points at current radar time represented under 
                            # the ego coordinate frame at front_cam_time.
                            radar_det_points_under_current_radar_frame_at_current_radar_time = Interface.RadarData(radar_time, i, current_pc, np.array(radar_cal['rotation']),
                                                            np.array(radar_cal['translation']), radar_token)
                            # radar_det_points_under_ego_frame_at_front_cam_time is the data structure which will store the radar det points under ego frame at front camera time, 
                            # but for now it is NOT 'radar det points under ego frame at front camera time' yet but just 'radar_det_points_under_current_radar_frame_at_current_radar_time'
                            radar_det_points_under_ego_frame_at_front_cam_time = radar_det_points_under_current_radar_frame_at_current_radar_time
                            # Transformation of position of radar det points from radar coordinate frame at current radar time to ego coordinate frame at front camera time
                            radar_det_points_under_ego_frame_at_front_cam_time.pc[:3, :] = trans_matrix_ego_from_radar_in_different_time.dot(
                                                                                np.vstack((radar_det_points_under_current_radar_frame_at_current_radar_time.pc[:3, :], 
                                                                                np.ones(radar_det_points_under_current_radar_frame_at_current_radar_time.pc.shape[1]))))[:3, :]
                            
                            # Calculate radar velocities information vx_comp, vy_comp(ego vehicle compensated velocity), for the radar detection points at current 
                            # radar time represented under the ego coordinate frame at front_cam_time. Beware there is only rotation for velocity rather than 
                            # translation when transformation from one coordinate system to another(there is no 'position' attribute for velocity, only 'orientation' attribute for velocity) 
                            velocities_under_current_radar_frame_at_current_radar_time = np.vstack((radar_det_points_under_current_radar_frame_at_current_radar_time.pc[8:10, :], 
                                                                                            np.zeros(radar_det_points_under_current_radar_frame_at_current_radar_time.pc.shape[1])))

                            # Beware the transformation_matrix of a Quaternion is only rotation matrix(cause Quaternion is only regarding rotation).
                            # Beware the rotation matrix is orthogonal matrix that inverse of orthogonal matrix equals to transpose of orthogonal matrix, thus here we use
                            # Quaternion(ref_pose_rec['rotation']).transformation_matrix.T rather than np.linalg.inv(Quaternion(ref_pose_rec['rotation']).transformation_matrix)
                            rotation_matrix_ego_from_radar_in_different_time = reduce(np.dot, [Quaternion(ref_pose_rec['rotation']).transformation_matrix.T, 
                                                                                Quaternion(current_pose_rec['rotation']).transformation_matrix, Quaternion(radar_cal['rotation']).transformation_matrix])
                            velocities_under_ego_frame_at_front_cam_time = np.dot(rotation_matrix_ego_from_radar_in_different_time[:, :3], velocities_under_current_radar_frame_at_current_radar_time)
                            radar_det_points_under_ego_frame_at_front_cam_time.pc[8:10, :] = velocities_under_ego_frame_at_front_cam_time[:2, :]

                            # change current radar time be as front_cam_time
                            radar_time = front_cam_time

                            # Accumulate all the transformed(in space and time synchronized) radar detection points until this loop.
                            if time_scan == 0:
                                accumulated_radar_points = radar_det_points_under_ego_frame_at_front_cam_time.pc
                            else:
                                accumulated_radar_points = np.hstack((accumulated_radar_points, radar_det_points_under_ego_frame_at_front_cam_time.pc))
                        else: # without interpolation in time(without time synchronization, still using radar data at current radar time directly)
                            radar_det_points_under_current_radar_frame_at_current_radar_time = Interface.RadarData(radar_time, i, current_pc, np.array(radar_cal['rotation']),
                                                            np.array(radar_cal['translation']), radar_token)
                            radar_det_points_under_ego_frame_at_current_radar_time = radar_det_points_under_current_radar_frame_at_current_radar_time
                            radar_det_points_under_ego_frame_at_current_radar_time.pc[:3, :] = ego_from_radar_at_current_radar_time.dot(
                                                                                np.vstack((radar_det_points_under_current_radar_frame_at_current_radar_time.pc[:3, :], 
                                                                                np.ones(radar_det_points_under_current_radar_frame_at_current_radar_time.pc.shape[1]))))[:3, :]
                            velocities_under_current_radar_frame_at_current_radar_time = np.vstack((radar_det_points_under_current_radar_frame_at_current_radar_time.pc[8:10, :], 
                                                                                            np.zeros(radar_det_points_under_current_radar_frame_at_current_radar_time.pc.shape[1])))
                            velocities_under_ego_frame_at_current_radar_time = np.dot(Quaternion(radar_cal['rotation']).transformation_matrix[:, :3], velocities_under_current_radar_frame_at_current_radar_time)
                            radar_det_points_under_ego_frame_at_current_radar_time.pc[8:10, :] = velocities_under_ego_frame_at_current_radar_time[:2, :]
                        
                            # Accumulate all the transformed(in space without time synchronized) radar detection points until this loop.
                            if time_scan == 0:
                                accumulated_radar_points = radar_det_points_under_ego_frame_at_current_radar_time.pc
                            else:
                                accumulated_radar_points = np.hstack((accumulated_radar_points, radar_det_points_under_ego_frame_at_current_radar_time.pc))

                        # Abort if there are no next time scans within number_of_radar_time_scans_for_accumulation time scans.
                        if radar_rec['prev'] == '':
                            break
                        else:
                            radar_rec = self.nusc.get('sample_data', radar_rec['prev'])
                    radar_data[sensor_name] = Interface.RadarData(radar_time, i, accumulated_radar_points, np.array(radar_cal['rotation']),
                                                            np.array(radar_cal['translation']), radar_token)

                else:
                    # Get radar calibration data(from ego vehicle to radar)
                    radar_cal = self.nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])
                    #Retrieve radar point cloud data from current sample data
                    file_name = osp.join(self.nusc.dataroot, radar_rec['filename'])
                    RadarPointCloud.disable_filters()  #disable filters(so extract all the detection points from RadarPointCloud)
                    current_pc = RadarPointCloud.from_file(file_name)
                    ##radar data structure
                    #radar data syncrhonization by linear interpolation
                    if self.config.getboolean('Preprocessor', 'ShouldInterpl'):
                        delta_time = (front_cam_time - radar_time)*1e-6
                        if self.loop_counter>1: 
                            assert delta_time>=0,'delta time cannot be negative'
                        # here ego vehicle speed compensated detection speed is used(current_pc[8,:], current_pc[9,:] are vx_comp
                        # and vy_comp) to calculate the correct position x and y(current_pc[0,:], current_pc[1,:]) at the front_cam_time
                        current_pc[0,:]+=(current_pc[8,:]*delta_time)
                        current_pc[1,:]+=(current_pc[9,:]*delta_time)  
                        radar_time = front_cam_time
                    temp = Interface.RadarData(radar_time, i, current_pc, np.array(radar_cal['rotation']),
                                                            np.array(radar_cal['translation']), radar_token)
                    
                    # Homogeneous transformation matrix from current radar coordinate frame to ego car frame.
                    car_from_current = transform_matrix(temp.translation, Quaternion(temp.rotation),inverse=False)
                    temp.pc[:3, :] = car_from_current.dot(np.vstack((temp.pc[:3, :], np.ones(temp.pc.shape[1]))))[:3, :]
                    
                    # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                    # point cloud.
                    # Using ego frame as reference frame  # Compensated velocity
                    velocities = np.vstack((temp.pc[8:10, :], np.zeros(temp.pc.shape[1])))
                    velocities = np.dot(Quaternion(temp.rotation).rotation_matrix, velocities)
                    temp.pc[8:10, :] = velocities[:2, :]
                    
                    radar_data[sensor_name] = temp


            self.front_radar_data = radar_data['front_radar_data']
            self.FL_radar_data = radar_data['FL_radar_data']
            self.FR_radar_data = radar_data['FR_radar_data']

            #Make arrays in data classes immutable
            self.arrayImmu()
            
            self.radar_data_gp = Interface.RadarDataGroup(radars=[self.front_radar_data, self.FL_radar_data, self.FR_radar_data],
                                                          colors=['cyan','lightsalmon','lawngreen'])
            
            ##Ground truth data structure
            self.ground_truth_assign(current_sd_rec['token'])
            
            #Logging
            self.log(front_cam_time)
            
            #figure information
            fig_info = dict()
            fig_info['scene_name'] = self.current_scene['name']
            fig_info['verbose'] = self.versbose
            if prev_timestamp==0:
                fig_info['elapse_time'] = 0
            else:
                fig_info['elapse_time'] = self.front_cam_data.time_stamp-prev_timestamp
            
            VisualPlot.render_sample(nusc=self.nusc, radar_data_gp=self.radar_data_gp, front_cam_data=self.front_cam_data, 
                                     host_data=self.host_data, fig_info=fig_info,
                                     out_path= self.out_path_for_figures + '/test_%d.png' % self.loop_counter)           
            
            if(current_sd_rec['next'] == ''):
                break
            else:
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec['next'])
                self.loop_counter+=1
                prev_timestamp = self.front_cam_data.time_stamp
                   

if __name__ == '__main__':
    dl = DataLoader()
    dl.run()