import logging
import os, sys
import time
import gym
import gym_tetris
import numpy as np


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
q_table = np.random.uniform(low=-1, high=1, size=(16*16, 6))
# 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
# epsilon_coefficient为贪心策略中的ε，取值范围[0,1]，取值越大，行为越随机
# 当epsilon_coefficient取值为0时，将完全按照q_table行动。故可作为训练模型与运用模型的开关值。
def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.0):
    # print(observation)
    next_state = observation
    epsilon = epsilon_coefficient * (0.99 ** episode)  # ε-贪心策略中的ε
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1, 2, 3 , 4 , 5])
    # -------------------------------------训练学习，更新q_table----------------------------------
    alpha = 0.2  # 学习系数α
    gamma = 0.99  # 报酬衰减系数γ
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, next_action])
    # -------------------------------------------------------------------------------------------
    return next_action, next_state
if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('Tetris-v0' if len(sys.argv)<2 else sys.argv[1])

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    video_path = '/tmp/random-agent-results'
    video_recorder = None
    # video_recorder = VideoRecorder(
    #     env, video_path, enabled=video_path is not None)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = RandomAgent(env.action_space)
    
    episode_count = 100
    max_steps = 200
    
    done = False

    last_time_steps = np.zeros(episode_count)

    # env.reset()
    # time.sleep(10)
    env.close()
    for i in range(episode_count):
        state = env.reset()
        episode_reward = 0
        q_table_cache = q_table
        for j in range(max_steps):

            action = np.argmax(q_table[state])
            ob, reward, done, _ = env.step(action)
            action, state = get_action(state, action, ob, reward, i, 0.5)  # 作出下一次行动的决策
            episode_reward += reward
            if done:
                np.savetxt("q_table.txt", q_table, delimiter=",")
                print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (i, j + 1, reward, last_time_steps.mean()))
                last_time_steps = np.hstack((last_time_steps[1:], [reward]))
                break
        q_table = q_table_cache
        episode_reward = -100
        print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
        last_time_steps = np.hstack((last_time_steps[1:], [reward]))  # 更新最近100场游戏的得分stack

        if (last_time_steps.mean() >= goal_average_steps):
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))
            env.close()
            sys.exit()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    # video_recorder.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")

