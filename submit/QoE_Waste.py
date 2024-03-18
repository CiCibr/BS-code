import numpy as np
import torch
device = 'cpu'
import sys
sys.path.append('./submit/results')
LOW_BITRATE_THRESHOLD = 1000
HIGH_BITRATE_THRESHOLD = 2000
# If there is no need to download, sleep for TAU time.
TAU = 1000.0  # ms
# max length of PLAYER_NUM
PLAYER_NUM = 5
# user retention threshold
RETENTION_THRESHOLD = 0.65
# fixed preload chunk num
PRELOAD_CHUNK_NUM = 4
last_chunk_bitrate = [-1, -1, -1, -1, -1, -1, -1]



def pad_action(act, act_param):
    params = [np.zeros((1,)), np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]
    params[act] = act_param
    return (act, params)


BIT_RATE0 = 'bit_rate0'
BIT_RATE1 = 'bit_rate1'
BIT_RATE2 = 'bit_rate2'
SLEEP = 'sleep'

ACTION_LOOKUP = {
    0: BIT_RATE0,
    1: BIT_RATE1,
    2: BIT_RATE2,
    3: SLEEP,
}

PARAMETERS_MIN = [
    np.array([0]),
    np.array([0]),
    np.array([0]),
    np.array([100])
]

PARAMETERS_MAX = [
    np.array([1]),
    np.array([1]),
    np.array([1]),
    np.array([1000])
]


class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0

        self.actor = torch.load('./submit/results/PDQN0_actor.pt')
        self.actor_param = torch.load('./submit/results/PDQN0_actor_param.pt')

    def Initialize(self):
        # Initialize your session or something
        self.buffer_size = 0

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # extract other features to help build the state
        # preload_size, buffer_size, video_chunk_counter, video_chunk_remain, user_retent_rate, play_timeline
        cur_player = Players[0]
        preload_size = cur_player.preload_size
        buffer_size = cur_player.buffer_size
        video_chunk_counter = cur_player.video_chunk_counter
        video_chunk_remain = cur_player.video_chunk_remain
        user_retent_rate = cur_player.user_retent_rate[
                           video_chunk_counter:video_chunk_counter + min(5, video_chunk_remain)]#只考虑当前块往后五个视频块的留存率
        play_timeline = cur_player.play_timeline
        if len(cur_player.download_chunk_bitrate) != 0:
            last_chunk_bitrate = cur_player.download_chunk_bitrate[-1]
        else:
            last_chunk_bitrate = 0
        user_retent_rate_fixed = np.arange(5, dtype=float)
        for i in range(5):
            if i < len(user_retent_rate):
                user_retent_rate_fixed[i] = float(user_retent_rate[i])
            else:
                user_retent_rate_fixed[i] = 0
        # expand the length of user_retent_rate to a fixed value

        state = np.array([delay, rebuf, video_size, end_of_video, play_video_id,
                          preload_size, buffer_size, video_chunk_counter, video_chunk_remain,
                          play_timeline, last_chunk_bitrate])
        for i in range(len(user_retent_rate_fixed)):
            state = np.append(state, user_retent_rate_fixed[i])

        # vote to decide the bitrate
        # vote_score = [0, 0, 1, 1, 2, 2, 1, 1, 1, 0, 0, 0]

        # use the trained model to decide the return value
        # state = np.array([delay, rebuf, video_size, end_of_video, play_video_id]).reshape(1, 5)
        state = state.astype(np.float32)
        # state = state.reshape(1, 16)
        # self.agent.device = 'cpu'
        # self.agent.action_max = self.agent.action_max.to('cpu')
        # self.agent.action_min = self.agent.action_min.to('cpu')
        # self.agent.action_parameter_max = self.agent.action_parameter_max.to('cpu')
        # self.agent.action_parameter_min = self.agent.action_parameter_min.to('cpu')
        # self.agent.action_parameter_range = self.agent.action_parameter_range.to('cpu')
        # self.agent.action_range = self.agent.action_range.to('cpu')
        state = torch.from_numpy(state).to(device)
        all_action_parameters = self.actor_param.forward(state)
        Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
        Q_a = Q_a.detach().cpu().data.numpy()
        action = np.argmax(Q_a)
        all_action_parameters = all_action_parameters.cpu().data.numpy()
        action_parameter_sizes = (1, 1, 1, 1)
        offset = np.array([action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
        action_param = all_action_parameters[offset:offset + action_parameter_sizes[action]]
        action = pad_action(action, action_param)

        bit_rate = action[0]
        # decide the download_video_id
        download_video_id = -1
        # flag = random.random()
        if Players[0].get_remain_video_num() != 0 and Players[0].get_chunk_counter() < int(
                Players[0].get_play_chunk()) + 3:  # downloading of the current playing video hasn't finished yet
            download_video_id = play_video_id #当前播放的视频还有未加载的视频块且即将加载的视频块是正在播放的视频块后三个块内，则后面加载当前视频的视频块
        else:
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                if Players[seq].get_chunk_counter() < PRELOAD_CHUNK_NUM and Players[
                    seq].get_remain_video_num() != 0:  # preloading hasn't finished yet
                    # 计算可能性：P(用户将观看将要预加载的块|用户从开头观看到当前播放得块)
                    start_chunk = int(Players[seq].get_play_chunk())
                    _, user_retent_rate = Players[seq].get_user_model()
                    cond_p = float(user_retent_rate[Players[seq].get_chunk_counter()]) / float(
                        user_retent_rate[start_chunk])
                    # if p > 留存率阈值
                    if cond_p > RETENTION_THRESHOLD:
                        download_video_id = play_video_id + seq
                        break
                #     if seq == min(len(Players), PLAYER_NUM) - 1:
                #         download_video_id = play_video_id + seq
                # else:
                #     bit_rate = -1
                #     sleep_time = 1000
        if download_video_id == -1:  # 无需加载, sleep for TAU time
            sleep_time = TAU
            download_video_id = play_video_id
        else:
            if bit_rate < 3:
                sleep_time = 0.0

            else:
                sleep_time = action[1][3][0]
        return download_video_id, bit_rate, sleep_time