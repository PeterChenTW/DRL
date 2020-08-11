import os

import numpy as np
import pandas as pd

from agent.ddqn import DDQN
from env.tic_tac_toe import OOXXRL

bot_chair = 1
game = OOXXRL(bot_chair)

bot = DDQN(game.action_dim, game.state_dim)

stats = bot.train(game)
df = pd.DataFrame(np.array(stats))
df.to_csv("log/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

exp_dir = 'models/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
