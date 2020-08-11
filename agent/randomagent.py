import random


class RandomBot:
    def predict(self, game):
        actions = game.legal_actions
        return random.sample(actions, 1)[0]
