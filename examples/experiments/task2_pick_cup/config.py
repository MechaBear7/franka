import os
import jax
import numpy as np
import jax.numpy as jnp

from serl_robot_infra.franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_robot_infra.franka_env.envs.relative_env import RelativeFrame
from serl_robot_infra.franka_env.envs.franka_env import DefaultEnvConfig

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from examples.experiments.config import DefaultTrainingConfig
from examples.experiments.task2_pick_cup.wrapper import PickCupEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        # * 腕部相机
        "wrist_1": {"serial_number": "115222071051", "dim": (1280, 720), "exposure": 10500},
        # * 侧面相机
        "side_policy_256": {"serial_number": "242422305075", "dim": (1280, 720), "exposure": 13000},
        # * 判定相机，Clone from side_policy_256
        "side_classifier": None,
        # * Demo 录制相机，Clone from side_policy_256
        "demo": None,
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img,
        "side_policy_256": lambda img: img[250:-150, 400:-500],  # 高度保留第250像素到-150像素，宽度保留第400像素到-500像素的图像
        "side_classifier": lambda img: img[390:-150, 420:-700],  # 高度保留第390像素到-150像素，宽度保留第420像素到-700像素的图像
        "demo": lambda img: img[50:-150, 400:-400]  # 高度保留第50像素到-150像素，宽度保留第400像素到-400像素的图像
    }

    TARGET_POSE = np.array([0.33, -0.15, 0.20, np.pi, 0, 0])  # TCP 完成任务的位姿
    RESET_POSE = np.array([0.61, -0.17, 0.22, np.pi, 0, 0])  # TCP 复位位姿
    ACTION_SCALE = np.array([0.08, 0.2, 1])  # 动作缩放
    RANDOM_RESET = True  # 是否随机复位
    DISPLAY_IMAGE = True  # 是否显示图像
    RANDOM_XY_RANGE = 0.02  # 随机 XY 范围
    RANDOM_RZ_RANGE = 0.03  # 随机 RZ 范围
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.03, 0.02, 0.01, 0.01, 0.3])  # TCP 位移限制
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.05, 0.05, 0.01, 0.01, 0.3])  # TCP 位移限制
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.005, 
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005, 
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02, 
        "rotational_Ki": 0,
    }  # 正常操作时的参数
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }  # 仅用于复位时的参数
    MAX_EPISODE_LENGTH = 100  # 最大步数
 

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Put the yellow banana to the green plate"
    octo_path = "/root/online_rl/octo_model/octo-small"

    def get_environment(
        self,
        fake_env=False,  # 是否使用假环境
        save_video=False,  # 是否保存视频
        classifier=False,  # 是否使用分类器
        stack_obs_num=1,  # 堆叠观测值的步数
    ):
        env = PickCupEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)  # spacemouse 干预 wrapper
        env = RelativeFrame(env)  # 坐标系转换 wrapper
        env = Quat2EulerWrapper(env)  # 将四元数转换为欧拉角 wrapper
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)  # 观测值展平 wrapper
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)  # 堆叠观测值，多步执行动作的 wrapper
        
        if classifier:
            # 加载分类器
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                def sigmoid(x):
                    return 1 / (1 + jnp.exp(-x))
                # * 我们期望在放置完毕后，夹爪抬升到一定高度，并且夹爪打开
                # 如果分类器预测成功，并且夹爪打开，并且TCP高度大于0.16，则奖励10.0
                if sigmoid(classifier(obs)[0]) > 0.9 and env.curr_gripper_pos > 0.5 and env.currpos[2] > 0.16:
                    return 10.0
                else:
                    return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        env = GripperPenaltyWrapper(env, penalty=-0.2)  # 如果夹爪动作过大，则进行惩罚

        return env
