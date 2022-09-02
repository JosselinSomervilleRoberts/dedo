"""
Environment that derives from DeformEnv and uses robots instead of anchors
for grasping and control. This class is experimental, so might only be
appropriate for expert users.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import os

import gym
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.bullet_manipulator import BaseManipulator, BulletManipulator
from ..utils.init_utils import get_preset_properties
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import DEFORM_INFO, ROBOT_INFO

from .deform_env import DeformEnv


def add_debug_pos(sim, pos, size=0.01, clr=[1,0,1]):
    sim.addUserDebugLine(pos-np.array([size,0,0]), pos+np.array([size,0,0]),
                         lineColorRGB=clr)
    sim.addUserDebugLine(pos-np.array([0,size,0]), pos+np.array([0,size,0]),
                         lineColorRGB=clr)

class DeformRobotEnv(DeformEnv):
    ORI_SIZE = 3 * 2  # 3D position + sin,cos for 3 Euler angles
    FING_DIST = 0.01  # default finger distance

    def __init__(self, args):
        super(DeformRobotEnv, self).__init__(args)
        act_sz = 3
        if self.food_packing:
            act_sz += DeformRobotEnv.ORI_SIZE
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Wrapped as DeformEnvRobot with act', self.action_space)

    @staticmethod
    def unscale_pos(act, unscaled):
        if unscaled:
            return act
        return act * DeformEnv.WORKSPACE_BOX_SIZE

    def load_objects(self, sim: bclient.BulletClient, args, debug: bool):
        # Debug flags
        self.disable_anchors = args.disable_anchors
        self.plot_trajectory = args.plot_trajectory
        self.n_slack = args.n_slack

        res = super(DeformRobotEnv, self).load_objects(sim, args, debug)
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        robot_info = ROBOT_INFO.get(args.robot_name + f'{self.num_anchors:d}', None)
        assert(robot_info is not None)  # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])

        # Flags override
        if args.override_fix_base:
            robot_info['use_fixed_base'] = True

        if debug:
            print('Loading robot from', robot_path)
        self.robot = BulletManipulator(
            sim, robot_path, control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]),
            global_scaling=robot_info['global_scaling'],
            use_fixed_base=robot_info['use_fixed_base'],
            rest_arm_qpos=robot_info['rest_arm_qpos'],
            left_ee_joint_name=robot_info.get('left_ee_joint_name', None),
            left_ee_link_name=robot_info.get('left_ee_link_name', None),
            left_fing_link_prefix='panda_hand_l_', left_joint_suffix='_l',
            left_rest_arm_qpos=robot_info.get('left_rest_arm_qpos', None),
            debug=debug)
        return res

    def make_anchors(self):
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        assert (preset_dynamic_anchor_vertices is not None)
        for i in range(self.num_anchors):  # make anchors
            anchor_pos = np.array(mesh[preset_dynamic_anchor_vertices[i][0]])
            if not np.isfinite(anchor_pos).all():
                print('anchor_pos not sane:', anchor_pos)
                input('Press enter to exit')
                exit(1)
            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id
            print(self.robot.info.robot_id, link_id)
            if not self.disable_anchors:
                self.sim.createSoftBodyAnchor(
                    self.deform_id, preset_dynamic_anchor_vertices[i][0],
                    self.robot.info.robot_id, link_id)

    def get_tgt_pos_detailed(self, action, unscaled=False):
        result = {}
        # Note: action is in [-1,1], so we unscale pos (ori is sin,cos so ok).
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_pos = DeformRobotEnv.unscale_pos(action[0, :3], unscaled)
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:]
        tgt_kwargs = {'ee_pos': tgt_pos, 'ee_ori': tgt_ee_ori,
                      'fing_dist': DeformRobotEnv.FING_DIST}
        if self.num_anchors > 1:  # dual-arm
            res = self.robot.get_ee_pos_ori_vel(left=True)
            left_ee_pos, left_ee_ori = res[0], res[1]
            left_tgt_pos = DeformRobotEnv.unscale_pos(action[1, :3], unscaled)
            #print("tgt_pos =", tgt_pos, "\tleft_tgt_pos =", left_tgt_pos)
            left_tgt_ee_ori = left_ee_ori if action.shape[-1] == 3 else \
                action[1, 3:]
            tgt_kwargs.update({'left_ee_pos': left_tgt_pos,
                               'left_ee_ori': left_tgt_ee_ori,
                                'left_fing_dist': DeformRobotEnv.FING_DIST})
            result["left_ee_pos"] = left_ee_pos
            result["left_ee_ori"] = left_ee_ori
            result["left_tgt_pos"] = left_tgt_pos
            result["left_tgt_ee_ori"] = left_tgt_ee_ori
        dist_xy, dist_ori = self.robot.base.get_plane_distance_and_ori_to_tgt(tgt_pos)
        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)

        if not self.robot.base.fixed:
            base_pos_xy = self.robot.base.get_pos()[:2]
            base_pos = np.array([base_pos_xy[0], base_pos_xy[1], 6.])

            # Computes the desired planar distance
            dz = tgt_pos[2] - base_pos[2]
            desired_dist_xy = np.sqrt(BaseManipulator.DESIRED_DIST**2 - dz**2)

            # Gets the planar direction from the base to the target ee.
            tgt_pos_xy = tgt_pos[:2]
            vect_base_to_tgt_xy = tgt_pos_xy - base_pos_xy
            dir_base_to_tgt_xy = np.array([0,1])
            norm_vect_base_to_tgt_xy = np.linalg.norm(vect_base_to_tgt_xy)
            if norm_vect_base_to_tgt_xy > 0:
                dir_base_to_tgt_xy = vect_base_to_tgt_xy / norm_vect_base_to_tgt_xy

            # Computes the desired position such that the arm length is equal to DESIRED_DIST
            # and the robot faces the target ee.
            result["tgt_base_pos_xy"] = tgt_pos_xy - desired_dist_xy * dir_base_to_tgt_xy
            result["tgt_base_ori"] = np.arctan2(vect_base_to_tgt_xy[0], vect_base_to_tgt_xy[1])

        result["ee_pos"] = ee_pos
        result["ee_ori"] = ee_ori
        result["tgt_pos"] = tgt_pos
        result["tgt_ee_ori"] = tgt_ee_ori
        result["dist_xy"] = dist_xy
        result["dist_ori"] = dist_ori
        result["tgt_qpos"] = tgt_qpos
        return result


    def do_action(self, action, unscaled=False, dt=1./240):
        KD = 1.2
        KP = 0.2
        max_diff = 0.02

        for sub_i in range(self.n_slack): # use > 1 if robot has trouble reaching the pose
            # Gets the target qpos (to move the arm)
            positions_dict = self.get_tgt_pos_detailed(action, unscaled)
            diff_qpos = self.robot.get_qpos() - positions_dict["tgt_qpos"]

            # Plot the target trajectory
            if sub_i == 0 and self.plot_trajectory:
                add_debug_pos(self.robot.sim, positions_dict["tgt_pos"], clr = [0,1,0])
            
            # Stops the loop if we are enar the objective
            if positions_dict["dist_xy"] < BaseManipulator.DESIRED_DIST and not (np.abs(diff_qpos) > max_diff).any(): break

            # Moves the base
            if not self.robot.base.fixed:
                # Gets the target base pos
                base_lin_speed, base_rot_speed = self.robot.base.control_get_speeds(positions_dict["tgt_base_pos_xy"], positions_dict["tgt_base_ori"])
                # It seems to work better without turning the base
                # self.robot.move_base(np.array([base_lin_speed[0], base_lin_speed[1], 0]), np.array([-base_rot_speed]))
                self.robot.move_base(np.array([base_lin_speed[0], base_lin_speed[1], 0]), np.array([base_rot_speed]))

            # Moves the arm
            self.robot.move_to_qpos(positions_dict["tgt_qpos"], mode=pybullet.POSITION_CONTROL, kp=KP, kd=KD)
            
            # Steps through simulation
            self.sim.stepSimulation()

        # Plots the trajectory of the ee
        if self.plot_trajectory:
            ee_pos, _, _, _ = self.robot.get_ee_pos_ori_vel()
            add_debug_pos(self.robot.sim, ee_pos, clr = [1,0,0])
            print(self.robot.base.get_ori())

    def make_final_steps(self):
        ee_pos, ee_ori, *_ = self.robot.get_ee_pos_ori_vel()
        final_action = np.hstack([ee_pos, ee_ori]).reshape(1, -1)
        if self.num_anchors > 1:  # dual-arm
            left_ee_pos, left_ee_ori, *_ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            final_left_action = np.hstack(
                [left_ee_pos, left_ee_ori]).reshape(1, -1)
            final_action = np.vstack([final_action, final_left_action])
        if self.args.debug:
            print('final_action', final_action)
        info = {'final_obs': []}
        for sim_step in range(DeformEnv.STEPS_AFTER_DONE):
            self.do_action(final_action, unscaled=True)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                next_obs, _ = self.get_obs()
                info['final_obs'].append(next_obs)
        return info

    def get_grip_obs(self):
        grip_obs = []
        ee_pos, _, ee_linvel, _ = self.robot.get_ee_pos_ori_vel()
        grip_obs.extend(ee_pos)
        grip_obs.extend((np.array(ee_linvel) / DeformEnv.MAX_OBS_VEL))
        if self.num_anchors > 1:  # EE pos, vel of left arm
            left_ee_pos, _, left_ee_linvel, _ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            grip_obs.extend(left_ee_pos)
            grip_obs.extend((np.array(left_ee_linvel) / DeformEnv.MAX_OBS_VEL))

        return grip_obs

    def get_reward(self):
        if self.food_packing:
            return self.get_food_packing_reward()
        else:
            return super(DeformRobotEnv, self).get_reward()

    def get_food_packing_reward(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        # rigid_ids[1] is the box, rigid_ids[2] is the can
        box_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[1])
        can_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[2])
        vertex_cent = np.mean(vertex_positions, axis=0)
        dist1 = np.linalg.norm(vertex_cent - box_pos)
        dist2 = np.linalg.norm(vertex_cent - can_pos)

        dist = np.mean([dist1, dist2])
        rwd = -1.0 * dist / DeformEnv.WORKSPACE_BOX_SIZE

        # Squish penalty (to protect the fruit)
        vertices = np.array(vertex_positions)
        relative_dist = np.linalg.norm(vertices - vertices[[0]], axis=1)

        current_shape = relative_dist[self.deform_shape_sample_idx]
        penalty_rwd = np.linalg.norm(current_shape - self.deform_init_shape)
        rwd = rwd + penalty_rwd
        return rwd
