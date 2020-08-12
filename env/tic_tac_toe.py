import random

import numpy as np

from agent.ddqn import DDQN


class TTT:
    # 執行 action 會獲勝的條件，需要有其中一組組合
    win = {0: ({1, 2}, {3, 6}, {4, 8}),
           1: ({0, 2}, {4, 7}),
           2: ({0, 1}, {5, 8}, {4, 6}),
           3: ({4, 5}, {0, 6}),
           4: ({3, 5}, {1, 7}, {0, 8}, {2, 6}),
           5: ({3, 4}, {2, 8}),
           6: ({7, 8}, {0, 3}, {2, 4}),
           7: ({6, 8}, {1, 4}),
           8: ({6, 7}, {2, 5}, {0, 4})}

    def __init__(self):
        self.reset()

    def reset(self):
        # 0: empty 1: o 2: x
        self.deck = [0 for _ in range(9)]
        self.legal_actions = [i for i in range(9)]
        self.cur_player = 1
        self.done = False
        self.winner = 0
        # index 1 -> player 1 history
        self.player_history = [(), [], []]

    def step(self, action):
        if self.done:
            print('game over!')
        if action not in self.legal_actions:
            print(f'error action: {action}')
        else:
            self.deck[action] = self.cur_player
            self.player_history[self.cur_player].append(action)
            self.legal_actions.remove(action)
            is_win = self.jugde_win(action)
            if is_win:
                self.finsh(self.cur_player)
            elif not self.legal_actions:
                self.finsh(None)
            else:
                self.to_next()

    def to_next(self):
        self.cur_player = 1 if self.cur_player == 2 else 2

    def finsh(self, winner):
        if winner:
            self.winner = self.cur_player
        self.done = True

    def jugde_win(self, action):
        for group in self.win[action]:
            if not group - set(self.player_history[self.cur_player]):
                return True
        return False

    def __str__(self):
        ans = '=' * 50 + '\n'
        ans += f'cur player: {self.cur_player}, end game: {self.done}, winner: {self.winner}\n'
        ans += '=' * 50 + '\n'
        count = 0
        for i in self.deck:
            if not i:
                ans += ' '
            else:
                ans += str(i)
            count += 1
            if count % 3:
                ans += '|'
            else:
                ans += '\n' + '-' * 6 + '\n'
        ans += '=' * 50
        return ans


class OOXXRL(TTT):
    def __init__(self, bot_chair):
        if bot_chair != 1 and bot_chair != 2:
            print(f'error chair: {bot_chair}')
        else:
            self.reset_count = 0
            self.action_dim = 9
            self.bot_chair = bot_chair
            self.state_dim = (1, 9)
            super().__init__()

    def reset(self):
        TTT.reset(self)
        if self.cur_player == self.bot_chair:
            return np.array([self.deck])
        else:
            self.step(self.opponent_action())
            return np.array([self.deck])

    def step(self, action):
        info = ''
        if action in self.legal_actions:
            TTT.step(self, action)
            if self.done:
                reward = self.compute_reward()
            else:
                TTT.step(self, self.opponent_action())
                reward = self.compute_reward()

            return np.array([self.deck]), reward, self.done, info
        else:
            reward = -1
            return np.array([self.deck]), reward, True, info

    def compute_reward(self):
        if self.winner == self.bot_chair:
            return 1
        elif self.winner == 0:
            return 0
        else:
            return -1

    def opponent_action(self):
        return random.sample(self.legal_actions, 1)[0]

    def ddqn_action(self):
        return self.bot.policy_action(np.array([self.deck]), is_test=True)

    def load_model(self):
        self.bot = DDQN(self.action_dim, self.state_dim)
        self.bot.load_weights('models/test_LR_0.00025_PER_dueling_for_o.h5')


if __name__ == '__main__':
    game = TTT()
    game.step(4)
    game.step(5)
    game.step(0)
    game.step(6)
    # game.step(8)
    print(game)
