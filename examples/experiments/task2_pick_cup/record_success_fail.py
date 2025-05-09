import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from examples.experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


# 如果按下空格键，则记录当前的 Transition 为成功，否则记录为失败
# Note: 需要说明，只有在最终完成任务时的那一个 Transition 才被记录为成功，这个标记为 step 级别，而不是 episode 级别
success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == 'Key.space':  # 如果按下空格键，则设置 success_key 为 True
            success_key = True
    except AttributeError:
        pass


def main(_):
    global success_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    obs, _ = env.reset()  # 重置环境，恢复到初始状态
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        
        # 如果环境返回了干预动作，则使用干预动作
        if "intervene_action" in info:
            actions = info["intervene_action"]
            # print(actions)

        # 创建一个字典来存储当前的过渡信息
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
            # obs, _ = env.reset()
        else:
            failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"

    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")


if __name__ == "__main__":
    app.run(main)
