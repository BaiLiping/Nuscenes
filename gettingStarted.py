from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import RadarPointCloud
import configparser

config = configparser.ConfigParser()
config.read('D:/Tech_Resource/Paper_Resource/Perception_R_C_Fusion_with_ShizhanWang_Project/code/config.ini')
nusc = NuScenes(version=config['Default']['Version'], dataroot=config['Default']['Dataroot'], 
                verbose=config.getboolean('Default','Versbose'))

nusc.list_scenes()
my_scene = nusc.scene[0]
my_sample_token = my_scene['first_sample_token']
first_sample = nusc.get('sample', my_sample_token)
loop = 0
while first_sample: 
    my_sample = nusc.get('sample', my_sample_token)
    #nusc.list_sample(my_sample['token'])
    sensor = 'RADAR_FRONT'
    radar_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    #camera_front_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    #nusc.render_sample_data(camera_front_data['token'])
    RadarPointCloud.disable_filters()
    nusc.render_sample_data(radar_front_data['token'], nsweeps=5)
    if(radar_front_data['timestamp']-1532402929662460 >0):
        print('loop: {}'.format(loop))
        print('front radar data at key frame timestamp: ', radar_front_data['timestamp'])
        plt.show()
    plt.close()
    if(my_sample['next'] == ''):
        break
    else:
        my_sample_token = my_sample['next']
        loop+=1
