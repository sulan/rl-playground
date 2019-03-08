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

    def __init__(self, actor_id, model, trajectory_length):
        super().__init__(actor_id, trajectory_length)
        self.model = model
        self.nb_actions = model.output._keras_shape[1]

    def get_action(self, observation):
        distribution = self.model.predict(observation.reshape(1, 1,-1))[0]
        action = np.random.choice(self.nb_actions, p = distribution)
        return action

def generate_episode_counter():
    """
    Returns with a simple counter function to used in episode index generation.
    """
    new_episode_index = -1
    def get_new_episode_index():
        nonlocal new_episode_index
        new_episode_index += 1
        return new_episode_index
    return get_new_episode_index

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

    def compile(self, optimizer, metrics = None):
        """
        Compiles an agent and the underlying models to be used for training
        and testing.

        # Arguments
        optimizer (`keras.optimizers.Optimizer` instance): The optimizer to
            be used during training.
        metrics (dict of functions `lambda y_true, y_pred: metric`): The
            metrics to run during training. The two keys should be 'pi' and 'V'.
        """
        if metrics is None:
            metrics = {
                'pi': [],
                'V': [],
                }
        # We never train the target model so the compilation parameters don't
        # matter
        self.model.compile(optimizer = 'sgd', loss = 'mse')

        # This corresponds to $\theta_{old}$ in the original PPO paper.
        self.cloned_model = clone_model(self.model)
        self.cloned_model.compile(optimizer = 'sgd', loss = 'mse')

        def clipped_loss(args):
            pi, pi_old, advantage, mask = args
            r = pi / pi_old
            clipped_r = K.clip(r, 1 - self.clipping_epsilon,
                               1 + self.clipping_epsilon)
            L_clip = K.minimum(r * advantage, clipped_r * advantage)
            L_clip = K.sum(L_clip * mask, axis = -1)
            # We should maximize L_clip
            return -L_clip

        def vf_loss(args):
            V, V_target = args
            return K.square(V - V_target)

        def entropy_loss(args):
            pi, = args
            # We should maximize the entropy (minimize negentropy)
            return K.sum(pi * K.log(pi), axis = -1)

        def surrogate_loss(args):
            """Linear combination of the above three."""
            L_clip, L_vf, L_entropy = args
            return (L_clip
                    + self.vfloss_coeff * L_vf
                    + self.entropy_coeff * L_entropy)

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

        clipped_loss_layer = Lambda(clipped_loss, output_shape = (1,),
                                    name = 'clipped_loss')(
            [pi, pi_old, advantage, mask])
        vf_loss_layer = Lambda(vf_loss, output_shape = (1,),
                               name = 'vf_loss')(
            [V, V_target])
        entropy_loss_layer = Lambda(entropy_loss, output_shape = (1,),
                                    name = 'entropy_loss')(
            [pi])
        # Many outputs:
        # - the 1st, 2nd and 3rd are the loss components
        # - the 4th and 5th are pi and V, for the metrics
        self.trainable_model = Model(
            inputs = ins + [pi_old, V_target, advantage, mask],
            outputs = [clipped_loss_layer, vf_loss_layer, entropy_loss_layer,
                       pi, V])

        loss = {
            'clipped_loss': lambda _, y_pred: y_pred,
            'vf_loss': lambda _, y_pred: y_pred,
            'entropy_loss': lambda _, y_pred: y_pred,
            self.model.output_names[0]: lambda _, y_pred: K.zeros_like(y_pred),
            self.model.output_names[1]: lambda _, y_pred: K.zeros_like(y_pred),
            }
        loss_weights = {
            'clipped_loss': 1,
            'vf_loss': self.vfloss_coeff,
            'entropy_loss': self.entropy_coeff,
            self.model.output_names[0]: 0,
            self.model.output_names[1]: 0,
        }
        metrics = {
            self.model.output_names[0]: metrics['pi'],
            self.model.output_names[1]: metrics['V'],
            }
        self.trainable_model.compile(optimizer = optimizer, loss = loss,
                                     loss_weights = loss_weights,
                                     metrics = metrics)

        self.compiled = True

    def create_actor(self, i):
        actor_model = Model(inputs = self.model.inputs,
                            outputs = self.model.outputs[0])
        actor = PolicyGradientActor(i, actor_model, self.trajectory_length)
        actor.get_new_episode_index = generate_episode_counter()
        return actor

    def backward(self, trajectories):
        assert self.compiled, 'PPOLearner must be compiled before training'
        # Without the ends (final state) of the trajectories
        # trajectories: list of (done, trajectory)
        states = sum(
            ([transition[0] for transition in trajectory[:-1]]
             for _, trajectory in trajectories), [])
        states = np.array(states)
        states.shape = (-1, 1) + states.shape[1:]
        # The states and the values at them at the end
        end_states = np.array(
            [trajectory[-1] for _, trajectory in trajectories])
        end_states.shape = (-1, 1) + end_states.shape[1:]
        _, end_values = self.cloned_model.predict(end_states)
        pi_old, V_old = self.cloned_model.predict(states)
        V_old.shape = (-1,)
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
                V_old[trajectory_start:
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
        dummy_target = [np.zeros((states.shape[0], 1))] \
            * len(self.trainable_model.output)
        history = self.trainable_model.fit([states, pi_old, V_targ_batch,
                                            advantage_batch, mask_batch],
                                           dummy_target,
                                           epochs = self.fit_epochs,
                                           verbose = 0)

        # Clone the new old model
        self.cloned_model.set_weights(self.trainable_model.get_weights())

        # FIXME: vf_loss is 0 for some reason
        # Throw away the dummy losses
        return [v for k, v in history.history.items()
                if k not in self.trainable_model.metrics_names[4:6]]

    @property
    def metrics_names(self):
        # Throw away the dummy losses (losses for pi and V)
        assert len(self.trainable_model.output_names) == 5
        pi_dummy_name = self.model.output_names[0]
        V_dummy_name = self.model.output_names[1]
        names = [name.replace(pi_dummy_name, 'pi').replace(V_dummy_name, 'V')
                 for i, name in enumerate(self.trainable_model.metrics_names)
                 if i not in (4, 5)]
        return names

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
