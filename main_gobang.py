import time

import numpy as np
import pandas as pd

from agent.ddqn import DDQN
from env.gobang import GobangRL

"""
train a DDQN model and save it for gobang
"""
bot_chair = 1
t = time.time()
game = GobangRL(bot_chair)

bot = DDQN(game.action_dim, game.state_dim)

stats = bot.train(game)

df = pd.DataFrame(np.array(stats))
df.to_csv(f'log/logs_{t}.csv', header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

export_path = f'models/{t}'

bot.save_weights(export_path)
