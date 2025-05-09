from typing import OrderedDict
import time
import copy
import requests
import numpy as np
import gymnasium as gym
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.franka_env import FrankaEnv


class PickCupEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """
        初始化相机。
        """
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side_policy_256"]
            elif cam_name == "demo":
                self.cap["demo"] = self.cap["side_policy_256"]
            else:
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap

    def reset(self, **kwargs):
        self._recover()  # 将机器人从错误状态恢复
        self._update_currpos()  # 更新当前位置到 self.currpos 等变量中
        self._send_pos_command(self.currpos)  # 发送位置命令
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)  # 更新参数
        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        # target[2] = self.config.RESET_POSE[2]
        self.interpolate_move(target, timeout=1)
        time.sleep(0.5)

        obs, info = super().reset(**kwargs)  # 调用父类 reset 方法

        self._send_gripper_command(1.0)  # 发送夹爪命令 夹爪打开
        time.sleep(1)
        self.success = False  # 初始化成功标志
        self._update_currpos()  # 更新当前位置到 self.currpos 等变量中
        obs = self._get_obs()  # 获取机器人当前状态的观测值

        return obs, info

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        执行重置的实际步骤应该在每个子类中实现。
        如果需要自定义重置过程，请重写此方法。
        """

        # 如果需要关节重置，则执行关节重置
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # 执行笛卡尔重置
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(-self.random_xy_range, self.random_xy_range, (2,))
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)
        time.sleep(1.0)

        # 切换到合规模式
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(-self.random_xy_range, self.random_xy_range, (2,))
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)  # 最后一位是夹爪动作
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        action = copy.deepcopy(action)
        grasp_action = action[..., -1]  # 夹爪动作

        # 将夹爪动作转换为 -1, 0, 1
        grasp_action = np.where(grasp_action > 0.5, 1, np.where(grasp_action < -0.5, -1, 0))
        action[..., -1] = grasp_action
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        if "intervene_action" in info:
            # 如果干预动作存在，则使用干预动作作为 action
            action = info["intervene_action"]

        # 如果夹爪动作过大，则惩罚
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.7) or (action[-1] > 0.5 and self.last_gripper_pos < 0.7):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]

        return observation, reward, terminated, truncated, info
