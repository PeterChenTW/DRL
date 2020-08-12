import numpy as np
import pandas as pd

from agent.ddqn import DDQN
from env.tic_tac_toe import OOXXRL

bot_chair = 2
game = OOXXRL(bot_chair)

bot = DDQN(game.action_dim, game.state_dim)
# bot.load_weights('models/test_LR_0.00025_PER_dueling.h5')

stats = bot.train(game)
df = pd.DataFrame(np.array(stats))
df.to_csv("log/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

export_path = 'models/test'

bot.save_weights(export_path)
