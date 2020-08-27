import copy
import math
import random
from collections import defaultdict as ddict

import numpy as np


class Gobang:
    def __init__(self, length=10):
        self.length = length
        self.reset()

    def reset(self):
        # 0: empty 1: o 2: x
        self.deck = np.zeros((self.length, self.length))
        self.legal_actions = [(i, j) for i in range(self.length) for j in range(self.length)]
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
            self.deck[action[0]][action[1]] = self.cur_player
            self.player_history[self.cur_player].append(action)
            self.legal_actions.remove(action)
            is_win = self.jugde_win(action)
            if is_win:
                self.finsh(self.cur_player)
            elif not self.legal_actions:
                self.finsh(None)
            else:
                self.to_next()

    def is_game_over(self):
        return self.done

    def to_next(self):
        self.cur_player = 1 if self.cur_player == 2 else 2

    def finsh(self, winner):
        if winner is not None:
            self.winner = self.cur_player
        self.done = True

    def jugde_win(self, action, deck=None):
        if deck is None:
            deck = self.deck
        # 判斷四個方向 有無五連珠 0度 45度 90度 135度
        x, y = action
        count = 0
        # 向左搜索
        for i in range(x + 1, self.length):
            if deck[i][y] == deck[x][y]:
                count += 1
            else:
                break
        # 向右搜索
        for i in range(x, 0, -1):
            if deck[i][y] == deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向下搜索
        for i in range(y + 1, self.length):
            if deck[x][i] == deck[x][y]:
                count += 1
            else:
                break
        # 向上搜索
        for i in range(y, 0, -1):
            if deck[x][i] == deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向右下搜索
        for i, j in zip(range(x + 1, self.length), range(y + 1, self.length)):
            if deck[i][j] == deck[x][y]:
                count += 1
            else:
                break
        # 向左上搜索
        for i, j in zip(range(x, 0, -1), range(y, 0, -1)):
            if deck[i][j] == deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向左下搜索
        for i, j in zip(range(x - 1, 0, -1), range(y + 1, self.length)):
            if deck[i][j] == deck[x][y]:
                count += 1
            else:
                break
        # 向右上搜索
        for i, j in zip(range(x, self.length), range(y, 0, -1)):
            if deck[i][j] == deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        return False

    def __str__(self):
        ans = '=' * self.length * 7 + '\n'
        ans += f'cur player: {self.cur_player}, end game: {self.done}, winner: {self.winner}\n'
        ans += f'legal action: {self.legal_actions}\n'
        ans += '=' * self.length * 7 + '\n'
        ans += '    ' + ''.join([f'  {i}   ' for i in range(self.length)]) + '\n'
        ans += '   |' + '-' * (self.length * 6 - 1) + '|\n'
        count = 0
        for i in self.deck:
            ans += f' {count} |'
            count += 1
            for j in i:
                if not j:
                    ans += '     '
                elif j == 1:
                    ans += '  o  '
                elif j == 2:
                    ans += '  x  '

                ans += '|'
            ans += '\n   |' + '-' * (self.length * 6 - 1) + '|\n'
        ans += '=' * self.length * 7
        return ans


class GobangRL(Gobang):
    def __init__(self, bot_chair):
        if bot_chair != 1 and bot_chair != 2:
            print(f'error chair: {bot_chair}')
        else:
            self.rl_len = 10
            self.action_dim = self.rl_len * self.rl_len
            self.bot_chair = bot_chair
            self.state_dim = (1, self.rl_len, self.rl_len)
            super().__init__(self.rl_len)

    def reset(self):
        Gobang.reset(self)
        if self.cur_player == self.bot_chair:
            return self.deck.reshape((1, self.rl_len, self.rl_len))
        else:
            self.step(self.opponent_action())
            return self.deck.reshape((1, self.rl_len, self.rl_len))

    def _action_index_to_cord(self, action_index):
        x = action_index // self.length
        y = action_index % self.length
        return x, y

    def step(self, ori_action):
        info = ''
        action = self._action_index_to_cord(ori_action)

        # if action in self.legal_actions:
        Gobang.step(self, action)
        if Gobang.is_game_over(self):
            reward = self.compute_reward()
        else:
            Gobang.step(self, self.opponent_action())
            reward = self.compute_reward()

        return self.deck.reshape((1, self.rl_len, self.rl_len)), reward, Gobang.is_game_over(self), info

    def sample_step(self, action):
        Gobang.step(self, action)

    def compute_reward(self):
        if self.winner == self.bot_chair:
            # print("win!!!!!!!!")
            # print(self)
            return 1
        elif self.winner == 0:
            if self.done:
                return 0.5
            return 0
        else:
            # print("lose!!!!!!!!")
            # print(self)
            return -1

    def opponent_action(self):
        # win to win
        for a in self.legal_actions:
            tmp_deck = self.deck.copy()
            tmp_deck[a[0]][a[1]] = 2
            if self.jugde_win(a, tmp_deck):
                return a
        # defense lose
        for a in self.legal_actions:
            tmp_deck = self.deck.copy()
            tmp_deck[a[0]][a[1]] = 1
            if self.jugde_win(a, tmp_deck):
                return a
        return random.sample(self.legal_actions, 1)[0]

    mcts_database = ddict(lambda: [0, 0])

    def mcts_action(self, times=10):
        # 0: total 1: win
        all_move_with_history = []
        for m in self.legal_actions:
            _room = copy.deepcopy(self)
            _room.sample_step(m)
            all_move_with_history.append([m, self.mcts_database[str(_room.deck)]])

        for n in range(times):
            _room = copy.deepcopy(self)
            # selection and expansion
            select_move = max(all_move_with_history,
                              key=lambda x: self._ucb_score(x[1][1], x[1][0], n + 1))[0]
            history = [str(_room.deck)]
            _room.sample_step(select_move)
            # simulation
            while not _room.done:
                if _room.cur_player == self.cur_player:
                    history.append(str(_room.deck))
                sample_move = random.sample(_room.legal_actions, 1)[0]
                _room.sample_step(sample_move)
            # back propagation
            for h in history:
                self.mcts_database[h][0] += 1
                if _room.winner == self.cur_player:
                    self.mcts_database[h][1] += 1

        final_move = max(all_move_with_history,
                         key=lambda x: x[1][1] / (x[1][0] + 1e-10))[0]

        return final_move

    def _ucb_score(self, win, total, n, c=1.414):
        _total = total + 1e-10
        return win / _total + math.sqrt(c * math.log(n) / _total)


if __name__ == '__main__':
    game = Gobang(10)
    while not game.done:
        print(game)
        action = input('your action: ')
        action = action.split(' ')
        action = tuple([int(i) for i in action])
        game.step(action)
    print(game)
