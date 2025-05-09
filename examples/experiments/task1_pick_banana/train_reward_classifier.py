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

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task1_pick_banana", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)  # 使用 PositionalSharding 进行设备分片
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=20000, include_label=True,)

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))  # 获取所有成功数据的路径
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))  # 加载成功数据
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2,}, device=sharding.replicate(),)  # 获取成功数据迭代器
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer( env.observation_space, env.action_space, capacity=50000, include_label=True,)  # 创建失败数据缓冲区  
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))  # 获取所有失败数据的路径
    for path in failure_paths:
        failure_data = pkl.load( open(path, "rb"))  # 加载失败数据
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size // 2,}, device=sharding.replicate(),)  # 获取失败数据迭代器

    print(f"failed buffer size: {len(neg_buffer)}")  # 打印失败数据缓冲区的大小
    print(f"success buffer size: {len(pos_buffer)}")  # 打印成功数据缓冲区的大小

    rng = jax.random.PRNGKey(0)  # 初始化随机数生成器
    rng, key = jax.random.split(rng)  # 将随机数生成器分成两个
    pos_sample = next(pos_iterator)  # 获取成功数据样本
    neg_sample = next(neg_iterator)  # 获取失败数据样本
    sample = concat_batches(pos_sample, neg_sample, axis=0)  # 合并成功和失败数据样本

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["observations"], config.classifier_keys,)  # 创建分类器

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
