import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from mplcursors import cursor
from functools import reduce
import copy


class VisualPlot:
    def __init__(self):
        pass
    @classmethod
    def render_sample(cls,
                      nusc: 'NuScenes',
                      radar_data_gp: 'RadarDataGroup',
                      front_cam_data: 'CameraData',
                      host_data:'HostData',
                      fig_info:dict,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      out_path: str = None) -> None:
        """
        Render radar and camera sample_data in sample along with annotations.
        
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        cls.colors=radar_data_gp.colors
        fig, axes = plt.subplots(1, 2, figsize=(16, 24))
        #left diagram to show front view
        ax1 = axes[0]
        cls.render_front_view(cls, nusc=nusc, radar_data_gp=radar_data_gp, front_cam_data=front_cam_data,
                              ax =ax1, use_flat_vehicle_coordinates=False, with_anns=False, with_radar=True)
        #right diagram to show bird view
        ax2 = axes[1]       
        
        cls.render_bird_view(cls, nusc=nusc, radar_data_gp=radar_data_gp, front_cam_data=front_cam_data, 
                             host_data=host_data, ax =ax2, use_flat_vehicle_coordinates=True)
        #figure information
        fig.text(0.01, 0.97, 'Figure information: ', weight='bold', color='green',fontsize = 15) 
        fig.text(0.01, 0.93, 'package: '+nusc.version, fontsize = 12) 
        fig.text(0.01, 0.9, 'scene name: '+fig_info['scene_name'], fontsize = 12) 
        fig.text(0.01, 0.87, 'timestamp: '+str(front_cam_data.time_stamp*1e-6)+'(s)', fontsize = 12)
        fig.text(0.01, 0.84, 'elapse time: : '+str((fig_info['elapse_time'])*1e-3)+'(ms)', fontsize = 12)
        
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if out_path is not None:
            plt.savefig(out_path,dpi=200)
        if fig_info['verbose']:    
            plt.show()

    
    def render_front_view(self,
                         nusc:'NuScenes',
                         radar_data_gp: 'RadarDataGroup',
                         front_cam_data: 'CameraData',
                         with_anns: bool = True,
                         with_radar=False,
                         box_vis_level: BoxVisibility = BoxVisibility.ANY,
                         axes_limit: float = 40,
                         ax: Axes = None,
                         out_path: str = None,
                         underlay_map: bool = True,
                         use_flat_vehicle_coordinates: bool = True) -> None:
        """
        Render sample data onto axis.
       
        :param with_anns: Whether to draw annotations.
        :param with_anns: Whether to draw radar detection points.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        """
        #Get boxes based on "base sensor"
        _, boxes, _ = nusc.get_sample_data(front_cam_data.sd_token, box_vis_level=box_vis_level)
        if with_anns:
            for box in boxes:
                c = np.array(nusc.explorer.get_color(box.name)) / 255.0
                box.render(ax, view=front_cam_data.intrinsic, normalize=True, colors=(c, c, c))
        if with_radar:
            #radar points already in vehicle frame, then to camera sensor frame
            #trans_r2v = transform_matrix(front_radar_data.translation,Quaternion(front_radar_data.rotation),inverse=False)
            trans_v2c = transform_matrix(front_cam_data.translation,Quaternion(front_cam_data.rotation),inverse=True)
            #red for Front, blue for FL, green for FR
            scatters={}
            for i in range(len(radar_data_gp.radars)):
                radar_data = radar_data_gp.radars[i]
                points = trans_v2c.dot(np.vstack((radar_data.pc[:3,:], np.ones(radar_data.pc.shape[1]))))[:3, :]
                points = view_points(points,front_cam_data.intrinsic,normalize=True)[:2,:]
                
                scatters[i] = ax.scatter(points[0, :], points[1, :], color=self.colors[i], s=3.0)
        #Add label
        def cursor_annotations(sel):
            sel.annotation.set_text('Cam coord:\nu: {:.2f} \nv: {:.2f} \nindex: {}'.format(sel.target[0], sel.target[1],sel.target.index))
            sel.annotation.get_bbox_patch().set(fc="gold", alpha=0.9)
    
        crs = cursor([scatters[0],scatters[1],scatters[2]], hover=False)
        crs.connect("add", cursor_annotations)    
        data = front_cam_data.image
        ax.imshow(data)
        
        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)
        ax.axis('off')
        ax.set_title('Front View')
        ax.set_aspect('equal')
        
    def render_bird_view(self,
                         nusc:'NuScenes',
                         radar_data_gp: 'RadarDataGroup',
                         front_cam_data: 'CameraData',
                         host_data:'HostData',
                         with_anns: bool = True,
                         box_vis_level: BoxVisibility = BoxVisibility.ANY,
                         axes_limit: float = 50,
                         ax: Axes = None,
                         out_path: str = None,
                         underlay_map: bool = True,
                         use_flat_vehicle_coordinates: bool = True) -> None:       
        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(host_data.rotation).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(host_data.rotation).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = vehicle_flat_from_vehicle
        else:
            viewpoint = np.eye(4)
            
        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                    'otherwise the location does not correspond to the map!'
            nusc.explorer.render_ego_centric_map(sample_data_token=front_cam_data.sd_token, axes_limit=axes_limit, ax=ax)
        velocities ={}
        scatters ={}
        for j in range(len(radar_data_gp.radars)):
            radar_data = radar_data_gp.radars[j]
            velocities[j] = copy.deepcopy(radar_data.pc[8:10, :]) # Compensated velocity
            velocities[j] = np.vstack((velocities[j], np.zeros(radar_data.pc.shape[1])))
            # Show point cloud.
            points = view_points(radar_data.pc[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(radar_data.pc[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 3.0 
            scatters[j] = ax.scatter(points[0, :], points[1, :], c=self.colors[j], s=point_scale)
                        
            # predicted point location if the same speed is kept for one second
            points_vel = view_points(radar_data.pc[:3, :] + velocities[j], viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 5 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            colors_rgba = scatters[j].to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=self.colors[j])
            
            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')
            spd_scal = 5
            ax.arrow(0,0,host_data.vx/spd_scal,host_data.vy/spd_scal,color='red')
        
        #Cursor labelling
        def cursor_annotations(sel):
            if sel.artist == scatters[0]:
                sel.annotation.set_text(
                    'X: {:.2f} m\nY: {:.2f} m\nVx: {:.2f} m/s\nVy: {:.2f} m/s\nindex: {}'.
                    format(sel.target[0], sel.target[1], velocities[0][0,sel.target.index], 
                            velocities[0][1,sel.target.index],sel.target.index))
                sel.annotation.get_bbox_patch().set(fc="gold", alpha=0.9)
            elif sel.artist == scatters[1]:
                sel.annotation.set_text(
                    'X: {:.2f} m\nY: {:.2f} m\nVx: {:.2f} m/s\nVy: {:.2f} m/s\nindex: {}'.
                    format(sel.target[0], sel.target[1], velocities[1][0,sel.target.index], 
                            velocities[1][1,sel.target.index],sel.target.index))
                sel.annotation.get_bbox_patch().set(fc="gold", alpha=0.9)   
            else:
                sel.annotation.set_text(
                    'X: {:.2f} m\nY: {:.2f} m\nVx: {:.2f} m/s\nVy: {:.2f} m/s\nindex: {}'.
                    format(sel.target[0], sel.target[1], velocities[2][0,sel.target.index], 
                            velocities[2][1,sel.target.index],sel.target.index))
                sel.annotation.get_bbox_patch().set(fc="gold", alpha=0.9)                  
        #hover option for hovering or clicking
        crs = cursor([scatters[0],scatters[1], scatters[2]], hover=False)
        crs.connect("add", cursor_annotations)
        
        # Get boxes in radar frame.
        boxes = nusc.get_boxes(front_cam_data.sd_token)
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(host_data.rotation).yaw_pitch_roll[0]
                box.translate(-np.array([host_data.x, host_data.y, host_data.z]))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            box_list.append(box)
       #TODO: Did this change make vehicle off the road a bit in some cases? To pay attentions continuously.     
       # _, boxes, _ = nusc.get_sample_data(radar_data_gp.radars[0].sd_token, box_vis_level=box_vis_level,
                                                #use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

        # Show boxes.
        if with_anns:
            for box in box_list:
                c = np.array(nusc.explorer.get_color(box.name)) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.axis('off')
        ax.set_title('Bird View')
        ax.set_aspect('equal')
        