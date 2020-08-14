import random

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

    def to_next(self):
        self.cur_player = 1 if self.cur_player == 2 else 2

    def finsh(self, winner):
        if winner:
            self.winner = self.cur_player
        self.done = True

    def jugde_win(self, action):
        # 判斷四個方向 有無五連珠 0度 45度 90度 135度
        x, y = action
        count = 0
        # 向左搜索
        for i in range(x + 1, self.length):
            if self.deck[i][y] == self.deck[x][y]:
                count += 1
            else:
                break
        # 向右搜索
        for i in range(x, 0, -1):
            if self.deck[i][y] == self.deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向下搜索
        for i in range(y + 1, self.length):
            if self.deck[x][i] == self.deck[x][y]:
                count += 1
            else:
                break
        # 向上搜索
        for i in range(y, 0, -1):
            if self.deck[x][i] == self.deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向右下搜索
        for i, j in zip(range(x + 1, self.length), range(y + 1, self.length)):
            if self.deck[i][j] == self.deck[x][y]:
                count += 1
            else:
                break
        # 向左上搜索
        for i, j in zip(range(x, 0, -1), range(y, 0, -1)):
            if self.deck[i][j] == self.deck[x][y]:
                count += 1
            else:
                break
        if count == 5:
            return True
        count = 0
        # 向左下搜索
        for i, j in zip(range(x - 1, 0, -1), range(y + 1, self.length)):
            if self.deck[i][j] == self.deck[x][y]:
                count += 1
            else:
                break
        # 向右上搜索
        for i, j in zip(range(x, self.length), range(y, 0, -1)):
            if self.deck[i][j] == self.deck[x][y]:
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
            self.action_dim = 5 * 5
            self.bot_chair = bot_chair
            self.state_dim = (5, 5)
            super().__init__(5)

    def reset(self):
        Gobang.reset(self)
        if self.cur_player == self.bot_chair:
            return self.deck
        else:
            self.step(self.opponent_action())
            return self.deck

    def step(self, ori_action):
        info = ''
        x = ori_action // self.length
        y = ori_action % self.length
        action = (x, y)
        if action in self.legal_actions:
            Gobang.step(self, action)
            if self.done:
                reward = self.compute_reward()
            else:
                Gobang.step(self, self.opponent_action())
                reward = self.compute_reward()
        else:
            reward = -1
            self.done = True
        return self.deck, reward, self.done, info

    def compute_reward(self):
        if self.winner == self.bot_chair:
            print("win!!!!!!!!")
            print(self)
            return 1
        elif self.winner == 0:
            if self.done:
                return 0.5
            return 0
        else:
            print("lose!!!!!!!!")
            print(self)
            return -1

    def opponent_action(self):
        return random.sample(self.legal_actions, 1)[0]


if __name__ == '__main__':
    game = Gobang(10)
    while not game.done:
        print(game)
        action = input('your action: ')
        action = action.split(' ')
        action = tuple([int(i) for i in action])
        game.step(action)
    print(game)
