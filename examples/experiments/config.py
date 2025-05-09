from abc import abstractmethod
from typing import List


class DefaultTrainingConfig:
    """Default training configuration. """

    agent: str = "drq"  # 使用 DRQ 算法
    max_traj_length: int = 100  # 最大轨迹长度
    batch_size: int = 256  # 256
    cta_ratio: int = 2
    discount: float = 0.97

    max_steps: int = 1000000  # 最大步数
    replay_buffer_capacity: int = 200000  # 重放缓冲区容量

    random_steps: int = 0  # 随机步数
    training_starts: int = 100  # 训练开始步数
    steps_per_update: int = 50  # 更新步数

    log_period: int = 10  # 日志周期
    eval_period: int = 2000  # 评估周期

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"  # 编码器类型
    demo_path: str = None  # 演示路径
    checkpoint_period: int = 0  # 检查点周期
    buffer_period: int = 0  # 缓冲区周期

    eval_checkpoint_step: int = 0  # 评估检查点步数
    eval_n_trajs: int = 5  # 评估轨迹数

    image_keys: List[str] = None  # 图像键
    classifier_keys: List[str] = None  # 分类器键
    proprio_keys: List[str] = None  # 关节位置键
    
    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"  # 设置模式

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
