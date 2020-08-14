import random

import numpy as np

from agent.ddqn import DDQN


class RandomBot:
    def predict(self, game):
        actions = game.legal_actions
        return random.sample(actions, 1)[0]


class DRLBot:
    def __init__(self):
        self.bot = DDQN(9, (1, 9))
        self.chair = 1

    def predict(self, game):
        if game.cur_player != self.chair:
            self.chair = game.cur_player
            self.update_model(self.chair)
        return self.bot.policy_action(np.array([game.deck]), is_test=True)

    def update_model(self, chair):
        if chair == 1:
            self.bot.load_weights('./models/1597218746.956845_LR_0.00025_PER_dueling_for_ttt_o_player.h5')
        elif chair == 2:
            self.bot.load_weights('./models/1597284912.602727_LR_0.00025_PER_dueling.h5')
