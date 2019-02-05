import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Input

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *

from a2c import A2C

class PolicyGradientActor(A2C.AbstractActor):
    """
    An A2C actor for the Proximal Policy Optimization algorithm.
    """

    # TODO: put softmax here

    def __init__(self, model, trajectory_length):
        super().__init__(trajectory_length)
        self.model = model
        self.nb_actions = model.output._keras_shape[1]

    def get_action(self, observation):
        distribution = self.model.predict(observation.reshape(1, 1,-1))[0]
        action = np.random.choice(self.nb_actions, p = distribution)
        return action

class PPOLearner(A2C.Learner):
    """
    An A2C learner for the Proximal Policy Optimization algorithm.

    arXiv:1707.06347
    """
    def __init__(self, model, trajectory_length, clipping_epsilon = 0.2,
                 gamma = 1, lam = 1, vfloss_coeff = 1, entropy_coeff = 1,
                 fit_epochs = 1):
        self.trajectory_length = trajectory_length
        self.clipping_epsilon = clipping_epsilon
        self.gamma = gamma
        self.lam = lam
        self.vfloss_coeff = vfloss_coeff
        self.entropy_coeff = entropy_coeff
        self.fit_epochs = fit_epochs
        # from a state, it estimaties the action and the value
        assert len(model.output) == 2, len(model.output)
        self.model = model
        self.nb_actions = model.output[0]._keras_shape[1]
        self.compiled = False

    def compile(self, optimizer, metrics = []):
        # We never train the target model so the compilation parameters don't
        # matter
        self.model.compile(optimizer = 'sgd', loss = 'mse')

        # This corresponds to $\theta_{old}$ in the original PPO paper.
        self.cloned_model = clone_model(self.model)
        self.cloned_model.compile(optimizer = 'sgd', loss = 'mse')

        def surrogate_loss(args):
            pi, pi_old, advantage, V, V_target, mask = args
            r = pi / pi_old
            clipped_r = K.clip(r, 1 - self.clipping_epsilon,
                               1 + self.clipping_epsilon)
            L_clip = K.minimum(r * advantage, clipped_r * advantage)
            entropy = K.sum(pi * K.log(pi), axis = -1)
            vf_loss = K.square(V - V_target)
            L_clip = K.sum(L_clip * mask, axis = -1)
            # Negative because Keras minimises the loss
            return -(L_clip + self.vfloss_coeff * vf_loss
                     + self.entropy_coeff * entropy)

        # action and value
        pi = self.model.output[0]
        V = self.model.output[1]
        action_shape = pi.shape[1:]
        # old action
        pi_old = Input(name = 'pi_old', shape = (action_shape))
        # target value
        V_target = Input(name = 'V_target', shape = (V.shape[1:]))
        # advantage
        advantage = Input(name = 'advantage', shape = (action_shape))
        # mask
        mask = Input(name = 'mask', shape = (action_shape))
        ins = [self.model.input] if type(self.model.input) is not list \
            else self.model.input

        loss_out = Lambda(surrogate_loss, output_shape = (1,))(
            [pi, pi_old, advantage, V, V_target, mask])
        self.trainable_model = Model(inputs = ins + [pi_old, V_target,
                                                     advantage, mask],
                                     outputs = loss_out)
        # trainable_model contains the loss
        loss = lambda _, y_pred: y_pred
        self.trainable_model.compile(optimizer = optimizer, loss = loss,
                                     metrics = metrics)

        self.compiled = True

    def create_actor(self, i):
        actor_model = Model(inputs = self.model.inputs,
                            outputs = self.model.outputs[0])
        return PolicyGradientActor(actor_model, self.trajectory_length)

    def backward(self, trajectories):
        assert self.compiled, 'PPOLearner must be compiled before training'
        # Without the ends of the trajectories
        # trajectory: list of (done, trajectory)
        states = sum(
            ([transition[0] for transition in trajectory[1][:-1]]
             for trajectory in trajectories), [])
        states = np.array(states)
        states.shape = (-1, 1) + states.shape[1:]
        # The states and the values at them at the end
        end_states = np.array([trajectory[1][-1] for trajectory in trajectories])
        end_states.shape = (-1, 1) + end_states.shape[1:]
        end_values = self.cloned_model.predict(end_states)[1]
        pi_old, V_old = self.cloned_model.predict(states)
        # Calculate advantages for each trajectory, starting from the end
        trajectory_start = 0
        advantage_batch = np.zeros((0, self.nb_actions))
        mask_batch = np.zeros((0, self.nb_actions))
        V_targ_batch = np.zeros((0, 1))
        for end_value, (done, trajectory) in zip(end_values, trajectories):
            _, actions, rewards = zip(*trajectory[:-1])
            actions = np.array(actions)
            rewards = np.array(rewards)
            # actual number of state transitions
            cur_trajectory_length = len(actions)
            # State values for the current trajectory (not including the state
            # at the end)
            V = np.append(
                V_old.flatten()[trajectory_start:
                                trajectory_start + cur_trajectory_length],
                0 if done else end_value)
            deltas = rewards + self.gamma * V[1:] - V[:-1]
            advantages = np.zeros((cur_trajectory_length, self.nb_actions))
            masks = np.zeros_like(advantages)

            next_advantage = 0
            V_targ = V[:-1]
            for delta, action, advantage, mask, V_targ_elem in zip(
                    deltas[::-1],
                    actions[::-1],
                    advantages[::-1],
                    masks[::-1],
                    V_targ[::-1]):
                advantage[action] = next_advantage = \
                    delta + self.gamma * self.lam * next_advantage
                mask[action] = 1
                V_targ_elem += next_advantage

            advantage_batch = np.append(advantage_batch, advantages, axis = 0)
            masks = np.zeros((cur_trajectory_length, self.nb_actions))
            masks[range(cur_trajectory_length), actions] = 1
            mask_batch = np.append(mask_batch, masks, axis = 0)
            V_targ.shape = (-1, 1)
            V_targ_batch = np.append(V_targ_batch, V_targ, axis = 0)

            trajectory_start += cur_trajectory_length
        mask_batch = np.array(mask_batch)
        dummy_target = np.zeros((states.shape[0], 1))
        self.trainable_model.fit([states, pi_old, V_targ_batch, advantage_batch,
                                  mask_batch],
                                 dummy_target,
                                 epochs = self.fit_epochs,
                                 verbose = 0)

        # Clone the new old model
        self.cloned_model.set_weights(self.trainable_model.get_weights())
        # TODO return metrics

    def get_config(self):
        config = {
            'trajectory_length': self.trajectory_length,
            'clipping_epsilon': self.clipping_epsilon,
            'gamma': self.gamma,
            'lam': self.lam,
            'vfloss_coeff': self.vfloss_coeff,
            'entropy_coeff': self.entropy_coeff,
            'fit_epochs': self.fit_epochs,
            'model': get_object_config(self.model)
        }
        if self.compiled:
            config['cloned_model'] = get_object_config(self.cloned_model)
        return config
