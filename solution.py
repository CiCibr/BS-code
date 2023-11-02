# PATH
# if you want to call your model ,the path setting is
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.discrete import Actor
from torch import nn, Tensor
import numpy as np
from typing import Union, Any, Dict, Tuple, Optional, Sequence
import random
# from ppo_discrete import Actor
import torch.nn.functional as F

LOW_BITRATE_THRESHOLD = 1000
HIGH_BITRATE_THRESHOLD = 2000
# If there is no need to download, sleep for TAU time.
TAU = 500.0  # ms
# max length of PLAYER_NUM
PLAYER_NUM = 5
# user retention threshold
RETENTION_THRESHOLD = 0.65
# fixed preload chunk num
PRELOAD_CHUNK_NUM = 4
last_chunk_bitrate = [-1, -1, -1, -1, -1, -1, -1]
pthfile_rainbow = "./log/Mmgc-v3/rainbow/policy.pth"
pthfile_sac = "./log/Mmgc-v0/discrete_sac/sac_policy.pth"
pthfile_ppo = "./log/Mmgc-v0/ppo/522policy_2.pth"
pthfile_ppo_others = "ppo.pth"
# NN_MODEL = "/home/team/"$YOUR TEAM NAME"/submit_pre/results/nn_model_ep_18200.ckpt" # model path settings
PPO = True
RAINBOW = False
PPO_others = False
SAC = False
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class MyNet(nn.Module):
    # 1D CNN + MLP
    def __init__(self, state_shape, output_dim, hidden_sizes=128,
                 device: Union[str, int, torch.device] = "cpu"):
        super(MyNet, self).__init__()
        self.input_dim = state_shape
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=2)
        self.model = nn.Sequential(*[
            self.conv1, nn.MaxPool1d(2), nn.ReLU(inplace=True),
            nn.Linear(9, hidden_sizes), nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes, hidden_sizes), nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes, np.prod(output_dim))
        ])
        self.device = device

    def forward(self, obs, state=None, info={}):
        # if not isinstance(obs, torch.Tensor):
        #     obs = torch.tensor(obs, dtype=torch.float)
        if self.device is not None:
            obs = torch.as_tensor(
                obs,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, 1, -1))
        logits = logits.view(batch, -1)
        return logits, state


class MyRecurrent(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))
        self.output_dim = action_shape

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: obs -> flatten -> logits.

        In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
        training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        obs = self.fc2(obs[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_others(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor_others, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, action_dim)

        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        s = s.view(-1, len(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0
        if PPO:
            # use the MLP structure
            # self.policy_net_ppo = torch.load(pthfile_ppo)
            # use the RNN structure
            # self.policy_net_ppo = MyRecurrent(layer_num=1, state_shape=5, action_shape=4, hidden_layer_size=128,
            #                                   device='cpu')
            mynet = MyNet(state_shape=16, output_dim=12, device='cpu')
            # mynet = MyRecurrent(1, state_shape=5, action_shape=4, hidden_layer_size=128, device='cpu')
            self.policy_net_ppo = Actor(mynet, action_shape=12, device='cpu')
            self.policy_net_ppo.load_state_dict(torch.load(pthfile_ppo))
        if RAINBOW:
            self.policy_net_d3qn = torch.load(pthfile_rainbow)
        if SAC:
            self.policy_net_sac = torch.load(pthfile_sac)
        if PPO_others:
            self.policy_net_ppo_others = Actor_others(16, 12, 128)
            self.policy_net_ppo_others.load_state_dict(torch.load(pthfile_ppo_others))
        # self.policy_net = torch.load(pthfile_rainbow).to(device)
        # self.policy_net.device = device
        # self.policy_net.model.device = device
        # self.policy_net.model.model.device = device
        # self.policy_net.model.Q.device = device
        # self.policy_net.model.V.device = device
        # self.policy_net.model_old.device = device
        # print(self.policy_net)
        # print('a')

    # Intial
    def Initialize(self):
        # Initialize your session or something
        self.buffer_size = 0

    # Define your algorithm
    # The args you can get are as follows:
    # 1. delay: the time cost of your last operation
    # 2. rebuf: the length of rebufferment
    # 3. video_size: the size of the last downloaded chunk
    # 4. end_of_video: if the last video was ended
    # 5. play_video_id: the id of the current video
    # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
    # 7. first_step: is this your first step?
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # extract other features to help build the state
        # preload_size, buffer_size, video_chunk_counter, video_chunk_remain, user_retent_rate, play_timeline
        cur_player = Players[0]
        preload_size = cur_player.preload_size
        buffer_size = cur_player.buffer_size
        video_chunk_counter = cur_player.video_chunk_counter
        video_chunk_remain = cur_player.video_chunk_remain
        user_retent_rate = cur_player.user_retent_rate[video_chunk_counter:video_chunk_counter+min(5, video_chunk_remain)]
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
        vote_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # use the trained model to decide the return value
        # state = np.array([delay, rebuf, video_size, end_of_video, play_video_id]).reshape(1, 5)
        state = state.astype(np.float32)
        state = state.reshape(1, 16)
        if PPO:
            # if use the MLP structure
            # logits, h = self.policy_net_ppo.actor(state)
            # if use the RNN structure
            logits, h = self.policy_net_ppo(state)
            logits = Tensor.cpu(logits).detach().numpy()
            vote_score[np.argmax(logits)] += 1
        if RAINBOW:
            logits, h = self.policy_net_d3qn.model(state)
            logits = torch.Tensor(logits.cpu())
            logits = Tensor.cpu(logits).detach().numpy()
            # logits = Tensor.cpu(logits).detach().numpy()
            # tmp = np.argmax(np.argmax(logits, axis=2))
            vote_score[np.argmax(np.argmax(logits, axis=2))] += 1
        if PPO_others:
            output = self.policy_net_ppo_others(torch.from_numpy(state))
            print(output)
            output = Tensor.cpu(output).detach().numpy()
            vote_score[np.argmax(output)] += 1
        if SAC:
            output, h = self.policy_net_sac.actor(state)
            output = Tensor.cpu(output).detach().numpy()
            vote_score[np.argmax(output)] += 1

        bit_rate = vote_score.index(max(vote_score))
        # decide the download_video_id
        download_video_id = -1
        # flag = random.random()
        if Players[0].get_remain_video_num() != 0 and Players[0].get_chunk_counter() < int(
                Players[0].get_play_chunk()) + 3:  # downloading of the current playing video hasn't finished yet
            download_video_id = play_video_id
        else:
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                if Players[seq].get_chunk_counter() < PRELOAD_CHUNK_NUM and Players[
                    seq].get_remain_video_num() != 0:  # preloading hasn't finished yet
                    # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)
                    start_chunk = int(Players[seq].get_play_chunk())
                    _, user_retent_rate = Players[seq].get_user_model()
                    cond_p = float(user_retent_rate[Players[seq].get_chunk_counter()]) / float(
                        user_retent_rate[start_chunk])
                    # if p > RETENTION_THRESHOLD, it is assumed that user will watch this chunk so that it should be preloaded.
                    if cond_p > RETENTION_THRESHOLD:
                        download_video_id = play_video_id + seq
                        break
                #     if seq == min(len(Players), PLAYER_NUM) - 1:
                #         download_video_id = play_video_id + seq
                # else:
                #     bit_rate = -1
                #     sleep_time = 1000
        if download_video_id == -1:  # no need to download, sleep for TAU time
            sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id  # the value of bit_rate and download_video_id doesn't matter
        else:
            if bit_rate < 3:
                sleep_time = 0
            else:
                sleep_time = (bit_rate - 2) * 100
                bit_rate = 0
        return download_video_id, bit_rate, sleep_time
