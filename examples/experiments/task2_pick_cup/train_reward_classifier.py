import glob
import os
# import sys
# sys.path.append("/home/mechabear7/codes/conrft")
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import TRAIN_CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_string("classifier_path", None, "Path to classifier checkpoint.")


def main(_):
    assert FLAGS.exp_name in TRAIN_CONFIG_MAPPING, 'Experiment folder not found.'
    config = TRAIN_CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)  # 使用 PositionalSharding 进行设备分片

    def create_transition_buffer(env, file_pattern, label, capacity):
        """创建并填充指定类型的transition缓冲区"""
        buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=capacity, include_label=True)
        
        data_dir = os.path.join(os.getcwd(), "classifier_data")
        file_paths = glob.glob(os.path.join(data_dir, file_pattern))
        
        for path in file_paths:
            with open(path, "rb") as f:
                data = pkl.load(f)
                for trans in data:
                    # 跳过包含图像数据的 transition
                    if "images" not in trans['observations']:
                        trans["labels"] = label
                        trans['actions'] = env.action_space.sample()
                        buffer.insert(trans)
        
        iterator = buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2}, device=sharding.replicate())
        return buffer, iterator

    # 创建成功和失败的缓冲区
    pos_buffer, pos_iterator = create_transition_buffer(env, "*success*.pkl", label=1, capacity=20000)
    neg_buffer, neg_iterator = create_transition_buffer(env, "*failure*.pkl", label=0, capacity=50000)

    print(f"success buffer size: {len(pos_buffer)}")  # 打印成功数据缓冲区的大小
    print(f"failed buffer size: {len(neg_buffer)}")  # 打印失败数据缓冲区的大小

    rng = jax.random.PRNGKey(0)  # 初始化随机数生成器
    rng, key = jax.random.split(rng)  # 将随机数生成器分成两个
    pos_sample = next(pos_iterator)  # 获取成功数据样本
    neg_sample = next(neg_iterator)  # 获取失败数据样本
    sample = concat_batches(pos_sample, neg_sample, axis=0)  # 合并成功和失败数据样本

    rng, key = jax.random.split(rng)
    # 创建分类器
    classifier = create_classifier(key, sample["observations"], config.classifier_keys, FLAGS.classifier_path)

    def data_augmentation_fn(rng, observations):
        # 数据增强函数，对图像进行随机裁剪
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit  # 使用 jax.jit 装饰器，将 train_step 函数编译为 JIT 函数
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["observations"], rngs={"dropout": key}, train=True)
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn({"params": state.params}, batch["observations"], train=False, rngs={"dropout": key})
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)  # 获取成功数据样本
        neg_sample = next(neg_iterator)  # 获取失败数据样本
        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)  # 合并成功和失败数据样本
        rng, key = jax.random.split(rng)  # 将随机数生成器分成两个
        obs = data_augmentation_fn(key, batch["observations"])  # 数据增强
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )  # 添加数据增强后的图像和标签

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    checkpoints.save_checkpoint(os.path.join(os.getcwd(), "classifier_ckpt/"), classifier, step=FLAGS.num_epochs, overwrite=True,)


if __name__ == "__main__":
    app.run(main)
