from agent.randomagent import RandomBot
from env.tic_tac_toe import TTT

players = ['', RandomBot(), RandomBot()]
game = TTT()
score = [0, 0, 0]

for _ in range(1000):
    while not game.done:
        action = players[game.cur_player].predict(game)
        game.step(action)
    score[game.winner] += 1
    game.reset()
print(score)
