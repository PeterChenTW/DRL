import random
from random import random, randrange

import keras.backend as K
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from utils.memory_buffer import MemoryBuffer
from utils.stats import gather_stats


class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = True
        self.dueling = True
        self.action_dim = action_dim
        self.state_dim = state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.2
        self.buffer_size = 20000
        #
        if len(state_dim) < 3:
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, self.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, self.with_per)

    def policy_action(self, s, is_test=False):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() > self.epsilon or is_test:
            return np.argmax(self.agent.predict(s)[0])
        else:
            return randrange(self.action_dim)

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if self.with_per:
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, q)
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(self, env):
        nb_episodes = 15000 * (48 + 12 + 2)
        batch_size = 128
        is_gather_stats = True
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()

            while not done:
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if self.buffer.size() > batch_size:
                    self.train_agent(batch_size)
                    self.agent.transfer_weights()
            # Gather stats every episode for plotting
            if is_gather_stats:
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if self.with_per:
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(self.agent.predict(new_state))
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        if self.with_per:
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)


class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.network(dueling)
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))

        # Determine whether we are dealing with an image input (Atari) or not
        if len(self.state_dim) > 2:
            inp = Input((self.state_dim[1:]))
            x = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
            x = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_initializer='he_normal')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
        else:
            x = Flatten()(inp)
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

        if dueling:
            # Have the network estimate the Advantage function as an intermediate layer
            x = Dense(self.action_dim + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                       output_shape=(self.action_dim,))(x)
        else:
            x = Dense(self.action_dim, activation='linear')(x)
        return Model(inp, x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        history = self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)
        # print(history.history['loss'])

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_dim) > 2:
            return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3:
            return np.expand_dims(x, axis=0)
        else:
            return x

    def save(self, path):
        if self.dueling:
            path += '_dueling'
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
