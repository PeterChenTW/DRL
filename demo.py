from agent.agents import DRLBot
from env.tic_tac_toe import TTT

game = TTT()
# players = ['', DRLBot(), 'Human']
players = ['', 'Human', DRLBot()]
while not game.done:
    print(game)
    if players[game.cur_player] == 'Human':
        action = int(input('your action index: '))
    else:
        action = players[game.cur_player].predict(game)
    game.step(action)
print(game)
