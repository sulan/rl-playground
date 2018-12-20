import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Input

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *

from a2c import A2C

class PPOActor(A2C.Actor):
    def __init__(self, model):
        self.model = model
        self.nb_actions = len(model.output._keras_shape[1])

    def get_action(self, observation):
        distribution = self.model.predict(observation.reshape(1,-1))
        action = np.random.choice(self.nb_actions, distribution)
        return action

class PPOLearner(A2C.Learner):
    def __init__(self):
        pass

    def compile(self, optimizer, metrics = [])
        # TODO finish this
        # We never train the target model so the compilation parameters don't
        # matter
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer = 'sgd', loss = 'mse')
        self.model.compile(optimizer = 'sgd', loss = 'mse')

        # TODO V-target
        # TODO remove this
        # if self.target_model_update < 1:
        #     updates = get_soft_target_model_updates(
        #         self.target_model, self.model, self.target_model_update)
        #     optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def surrogate_loss(args):
            args = theta, theta_old, advantage = args
            r = theta / theta_old
            clipped_r = K.clip(r, 1 - self.clipping_epsilon,
                               1 + self.clipping_epsilon)
            L_clip = K.minimum(r * advantage, clipped_r * advantage)
            return L_clip

        # TODO create trainable model
        action = self.model.output[0]
        V = self.model.output[1]
        # TODO check shape
        action_pred = Input(name = 'action_pred', shape = (action.shape))
        ins = [self.model.input] if type(self.model.input) is not list \
            else self.model.input

        self.trainable_model = None

# class PPO(Agent):
#     def __init__(self, model, num_actors, clipping_epsilon = 0.2, **kwargs):
#         super().__init__(**kwargs)
#         self.num_actors = num_actors
#         self.clipping_epsilon = clipping_epsilon
#         # action and value
#         assert len(model.output) == 2, len(model.output)
#         self.model = model
#         self.reset_states()

#     def get_config(self):
#         config = super().get_config()
#         config['num_actors'] = self.num_actors
#         config['clipping_epsilon'] = self.clipping_epsilon

#     def compile(self, optimizer, metrics = []):
#         # We never train the target model so the compilation parameters don't
#         # matter
#         self.target_model = clone_model(self.model)
#         self.target_model.compile(optimizer = 'sgd', loss = 'mse')
#         self.model.compile(optimizer = 'sgd', loss = 'mse')

#         # TODO V-target
#         # TODO remove this
#         # if self.target_model_update < 1:
#         #     updates = get_soft_target_model_updates(
#         #         self.target_model, self.model, self.target_model_update)
#         #     optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

#         def surrogate_loss(args):
#             args = theta, theta_old, advantage = args
#             r = theta / theta_old
#             clipped_r = K.clip(r, 1 - self.clipping_epsilon,
#                                1 + self.clipping_epsilon)
#             L_clip = K.minimum(r * advantage, clipped_r * advantage)
#             return L_clip

#         # TODO create trainable model
#         action = self.model.output[0]
#         V = self.model.output[1]
#         # TODO check shape
#         action_pred = Input(name = 'action_pred', shape = (action.shape))
#         ins = [self.model.input] if type(self.model.input) is not list \
#             else self.model.input

#         self.trainable_model = None

#         self.compiled = True

#     def forward(self, observation):
#         # TODO Select an action
#         action = None

#         # Book-keeping
#         self.recent_observation = observation
#         self.recent_action = action

#         return action

#     def backward(self, reward, terminal):
#         pass

#     def reset_states(self):
#         self.recent_action = None
#         self.recent_observation = None
#         if self.compiled:
#             self.model.reset_states()
#             self.target_model.reset_states()
