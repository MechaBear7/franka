"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate([cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1)

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 1.0
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class FrankaEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE  # 是否显示图像
        self.gripper_sleep = config.GRIPPER_SLEEP  # 夹爪动作间隔时间
        self.config = config

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate([config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])])
        # self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        # reset the robot joint every 200 cycles
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # 每 200 个 episode 重置一次机器人关节

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(config.ABS_POSE_LIMIT_LOW[:3], config.ABS_POSE_LIMIT_HIGH[:3], dtype=np.float64)  # 机器人末端执行器的位置限制
        self.rpy_bounding_box = gym.spaces.Box(config.ABS_POSE_LIMIT_LOW[3:], config.ABS_POSE_LIMIT_HIGH[3:], dtype=np.float64)  # 机器人末端执行器的姿态限制
        # Action/Observation Space
        self.action_space = gym.spaces.Box(np.ones((7,), dtype=np.float32) * -1, np.ones((7,), dtype=np.float32))

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat，代表机器人末端执行器的位置和姿态
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),  # 机器人末端执行器的速度，[:3]为线速度，[3:]为角速度
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),  # 机器人夹爪的开合状态，-1表示夹爪关闭，1表示夹爪打开
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),  # 机器人末端执行器的力
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),  # 机器人末端执行器的力矩
                    }
                ),
                # "images": gym.spaces.Dict(
                #     {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                #                 for key in config.REALSENSE_CAMERAS}
                # ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8) if '256' in key
                        else gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0  # reset 次数

        if fake_env:  # 如果是虚拟环境，则直接返回，不执行以下代码
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:  # 是否设置负载
            input("Put arm into programing mode and press enter.")
            requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        if not fake_env:  # 如果不是虚拟环境，开启键盘监听
            from pynput import keyboard
            self.terminate = False

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized Franka")

    def init_cameras(self, name_serial_dict=None):
        """
        初始化相机，保存到 self.cap 中
        """
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap

    def close_cameras(self):
        """
        关闭 self.cap 中的所有相机
        """
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        # 将机器人末端执行器的位置限制在安全范围内
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (np.clip(np.abs(euler[0]), self.rpy_bounding_box.low[0], self.rpy_bounding_box.high[0]))  # 将第一个欧拉角限制在安全范围内

        euler[1:] = np.clip(euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:])  # 将欧拉角限制在安全范围内
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()  # 将欧拉角转换为四元数

        return pose

    def _send_pos_command(self, pos: np.ndarray):
        """
        发送位置命令到机器人，机器人会根据位置命令移动到目标位置。
        """
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """
        发送夹爪动作命令到机器人。
        """
        # print(pos, self.curr_gripper_pos)
        if mode == "binary":
            if (pos <= -0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
            # if (pos <= -0.5) and (self.curr_gripper_pos > 0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):
                requests.post(self.url + "close_gripper")  # 发送夹爪关闭命令
            elif (pos >= 0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
            # elif (pos >= 0.5) and (self.curr_gripper_pos < 0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):
                requests.post(self.url + "open_gripper")  # 发送夹爪打开命令
            else:
                return
            self.last_gripper_act = time.time()  # 更新夹爪上次动作时间
            time.sleep(self.gripper_sleep)  # 等待夹爪动作完成
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _recover(self):
        """
        将机器人从错误状态恢复。
        """
        requests.post(self.url + "clearerr")

    def _update_currpos(self):
        """
        获取机器人和夹爪的最新状态并更新到 self.currpos 等变量中。
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """
        使用线性插值移动机器人到目标位置。
        """
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        执行 reset 的具体步骤应该在每个子类中实现。
        如果需要自定义 reset 过程，请重写此方法。
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
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

    def _get_obs(self) -> dict:
        """
        获取机器人当前状态的观测值。
        """
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def compute_reward(self, obs) -> bool:
        """
        这个函数计算当前状态与目标状态的差异，并根据差异的大小返回一个布尔值，表示是否达到目标状态。
        """
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """
        从 realsense 相机获取图像。
        """
        images = {}
        display_images = {}
        full_res_images = {}  # 存储全分辨率裁剪图像的字典
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(cropped_rgb, self.observation_space["images"][key].shape[:2][::-1])
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                # Store the full resolution cropped image
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(f"{key} camera frozen. Check connect, then press enter to relaunch...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'

                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)  # 将动作限制在动作空间内
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (Rotation.from_euler("xyz", action[3:6] * self.action_scale[1]) * Rotation.from_quat(self.currpos[3:])).as_quat()  # 将欧拉角转换为四元数

        # action[6] = 0.0
        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)  # 发送夹爪动作
        self._send_pos_command(self.clip_safety_box(self.nextpos))  # 发送位置命令

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def reset(self, joint_reset=False, **kwargs):
        """
        重置机器人。
        """
        self.last_gripper_act = time.time()  # 更新夹爪上次动作时间
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()  # 恢复机器人
        self.go_to_reset(joint_reset=joint_reset)  # 移动到 reset 位置
        self._recover()  # 恢复机器人
        self.curr_path_length = 0  # 当前路径长度

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def close(self):
        """
        关闭键盘监听，关闭相机，关闭图像显示
        """
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
