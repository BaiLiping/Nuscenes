from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import RadarPointCloud


nusc = NuScenes(version='v1.0-mini', dataroot=r'D:\Tech_Resource\Paper_Resource\Dataset\nuscenes\v1.0-mini', verbose=True)

sensor_1 = 'RADAR_FRONT'
sensor_2 = 'CAM_FRONT'
sensor_3 = 'LIDAR_TOP'


nusc.list_scenes()
my_scene = nusc.scene[0]

"""----------------- Show the first sample data --------------------"""
first_sample_token = my_scene['first_sample_token']
first_sample = nusc.get('sample', first_sample_token)
radar_front_data = nusc.get('sample_data', first_sample['data'][sensor_1])
camera_front_data = nusc.get('sample_data', first_sample['data'][sensor_2])
lidar_top_data = nusc.get('sample_data', first_sample['data'][sensor_3])
nusc.render_sample_data(camera_front_data['token'])
nusc.render_sample_data(radar_front_data['token'])
nusc.render_sample_data(lidar_top_data['token'])

nusc.render_pointcloud_in_image(first_sample['token'], pointsensor_channel = sensor_3, render_intensity=True)
nusc.render_pointcloud_in_image(first_sample['token'], pointsensor_channel = sensor_1)

plt.show()


"""----------------- Show the second sample data --------------------"""
my_sample_token = first_sample['next']
my_sample = nusc.get('sample', my_sample_token)
nusc.list_sample(my_sample['token'])
radar_front_data = nusc.get('sample_data', my_sample['data'][sensor_1])
camera_front_data = nusc.get('sample_data', my_sample['data'][sensor_2])
lidar_top_data = nusc.get('sample_data', first_sample['data'][sensor_3])
nusc.render_sample_data(camera_front_data['token'])
RadarPointCloud.disable_filters()
nusc.render_sample_data(radar_front_data['token'])
nusc.render_sample_data(lidar_top_data['token'])

nusc.render_pointcloud_in_image(first_sample['token'], pointsensor_channel = sensor_3, render_intensity=True)
nusc.render_pointcloud_in_image(first_sample['token'], pointsensor_channel = sensor_1)
RadarPointCloud.default_filters()

plt.show()