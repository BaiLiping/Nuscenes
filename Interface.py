from dataclasses import dataclass, is_dataclass
import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import Box
from typing import Tuple, List


@dataclass(frozen=True)
class HostData():
    '''
    pos in global frame and others in vehicle frame
    '''
    time_stamp:int
    x:float
    y:float
    z:float
    vx:float
    vy:float
    vz:float
    ax:float
    ay:float
    az:float
    yaw_rate:float
    rotation:object #??

@dataclass(frozen=True)
class CameraData():
    time_stamp:int
    cam_id:int        #0: Front 1:FL 2:FR 3:RL 4:RR 5:Rear
    image:Image
    intrinsic:np.ndarray
    rotation:np.ndarray
    translation:np.ndarray
    sd_token:str
    
@dataclass(frozen=True)
class RadarData():
    time_stamp:int
    radar_id:int      #0: Front 1:FL 2:FR 3:RL 4:RR
    pc:np.ndarray     #point cloud positions and compensated velocities on vehicle frame (initially on sensor frame)
                      #Rest still on sensor frame
    rotation:np.ndarray
    translation:np.ndarray
    sd_token:str

@dataclass(frozen=True)
class RadarDataGroup():
    radars:List[RadarData]
    colors:List

@dataclass(frozen=True)
class Bbox():
    id:int
    x:float
    y:float
    z:float
    vx:float
    vy:float
    vz:float
    orien:float  #degree
    w:float
    l:float
    h:float
    categr:str
    #convert the object to a list
    def to_list(self):
        return [self.id, self.x, self.y, self.z, self.vx, self.vy, self.vz, self.orien, self.w, self.l, self.h, self.categr]

#nested dataclass  
@dataclass(frozen=True)
class GroundTruth():
    #EXAMPLE->label: nan, score: nan, xyz: [373.26, 1130.42, 0.80], wlh: [0.62, 0.67, 1.64], rot axis: [0.00, 0.00, -1.00], ang(degrees): 21.09, ang(rad): 0.37, vel: nan, nan, nan, name: human.pedestrian.adult, token: ef63a697930c4b20a6b9791f423351da
    #xyz global frame
    bboxes:List[Bbox]
    @property
    def total_bbox(self):
        return len(self.bboxes)
  

@dataclass(frozen=True)
class ObjectList():
    time_stamp:int
    object_array:np.ndarray