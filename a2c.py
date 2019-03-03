from copy import deepcopy

from keras.callbacks import History

from rl.callbacks import CallbackList

class A2C:
    """
    Implements an Advantage Actor Critic agent with similar API to
    rl.core.Agent.
    """

    # TODO processor
    # TODO random steps
    # TODO action repetition
    # TODO callbacks, history

    class Actor:
        """
        Actor of the Advantage Actor Critic algorithm, ie. the part of the
        implementation that interacts with (an instance of) the environment.
        """

        def get_action(self, observation):
            """
            Choose an action for the current observation.
            """
            raise NotImplementedError()

        def build_trajectory(self, env, max_trajectory_length, callbacks):
            """
            Interacts with the environment for trajectory_length steps.

            Stops when the episode ends or when max_trajectory_length is
            reached.

            Saves the trajectory

            # Arguments:
            env: The environment to interact with.
            max_trajectory_length (integer): The maximum length of the returned
                trajectory (which is shorter at the end of the training.)
            callbacks (`rl.callbacks.CallbackList`): Callbacks during training.

            # Returns:
                A pair of boolean and list values. The former indicates that
                the trajectory resulted in episode termination, and the latter
                is the trajectory (observation, action, reward) tuples.
            """
            raise NotImplementedError()

        def get_trajectory(self):
            """
            Returns the trajectory generated by (and saved by)
            `build_trajectory`.

            May delete the saved trajectory.

            # Returns:
                A pair of boolean and list values. The former indicates that
                the trajectory resulted in episode termination, and the latter
                is the trajectory (observation, action, reward) tuples.
            """
            raise NotImplementedError()

        def get_new_episode_index(self):
            """
            Returns with an index for the new episode.

            This index should differentiate between episodes generated by the
            different Actors of the same learning agent.

            # Returns:
                A new nonnegative index (integer) value that is different from
                the previous ones.
            """
            raise NotImplementedError()

        def reset(self):
            """
            Resets the actor.
            """
            pass


    class AbstractActor(Actor):
        """
        An A2C Actor that builds up a multistep trajectory
        """
        def __init__(self, actor_id, trajectory_length):
            self.actor_id = actor_id
            self.episode = None
            self.episode_step = 0
            self.trajectory_length = trajectory_length
            self.last_observation = None
            self.trajectory = None
            self.done = None
            self.episode_reward = None

        def build_trajectory(self, env, max_trajectory_length, callbacks):
            assert max_trajectory_length > 0, \
                'build_trajectory assumes at least one step'
            self.trajectory = []
            self.done = False
            if self.last_observation is None:
                # Start new episode
                self.last_observation = deepcopy(env.reset())
                self.episode = self.get_new_episode_index()
                self.episode_reward = 0
                self.episode_step = 0

            for _ in range(min(self.trajectory_length, max_trajectory_length)):
                callbacks.on_step_begin(self.episode_step)
                action = self.get_action(self.last_observation)
                observation, reward, self.done, info = env.step(action)
                observation = deepcopy(observation)
                self.trajectory.append((self.last_observation, action, reward))
                self.last_observation = observation

                step_logs = {
                    'actor': self.actor_id,
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': self.episode,
                    'info': info,
                    }
                callbacks.on_step_end(self.episode_step, step_logs)
                self.episode_step += 1
                self.episode_reward += reward

                if self.done:
                    episode_logs = {
                        'actor': self.actor_id,
                        'episode_reward': self.episode_reward,
                        'nb_episode_steps': self.episode_step,
                        }
                    callbacks.on_episode_end(self.episode, episode_logs)
                    break
            self.trajectory.append(self.last_observation)
            if self.done:
                self.last_observation = None
            return self.done, self.trajectory

        def get_trajectory(self):
            """
            Returns the trajectory generated by (and saved by)
            `build_trajectory`.

            Deletes the saved trajectory.

            # Returns:
                A pair of boolean and list values. The former indicates that
                the trajectory resulted in episode termination, and the latter
                is the trajectory (observation, action, reward) tuples.
            """
            assert self.done is not None, 'No trajectory built.'
            return self.done, self.trajectory

        def reset(self):
            self.last_observation = None
            self.done = None
            self.trajectory = None
            self.episode_reward = None


    class Learner:
        """
        Learner of the Advantage Actor Critic algorithm, ie. the part of the
        implementation that calculates the new parameters.
        """
        def backward(self, trajectories):
            """
            Update the parameters based on a single trajectory.
            """
            raise NotImplementedError()

        def create_actor(self, i):
            """
            Creates a new instance of the appropriate Actor subclass

            # Arguments:
            i (integer): The index of the actor to be created. Can be used to
                implement different exploration strategies, for example.
            """
            raise NotImplementedError()

    def __init__(self, learner, num_actors = 1):
        self.training = False
        self.learner = learner
        self.num_actors = num_actors
        self.step = 0
        self.compiled = False

    def compile(self, optimizer, metrics = None):
        """
        Compiles the A2C agent and the underlying model via the learner.

        # Arguments:
        optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be
            used during training.
        metrics (list of functions `lambda y_true, y_pred: metric`): The
            metrics to run during training.
        """
        metrics = [] if metrics is None else metrics
        self.learner.compile(optimizer, metrics)
        self.compiled = True

    def fit(self, env_factory, nb_steps, action_repetition = 1,
            callbacks = None):
        """
        Trains the agent on the given environment.

        # Arguments:
        env_factory (`lambda integer: env`): A factory function that gives an
            new environment instance. Its parameter is the index of the
            instance.
        nb_steps (integer): Number of training steps to be performed.
        action_repetition (integer): Number of times the agent repeats the same
            action without observing the environment again. Setting this to a
            value > 1 can be useful if a single action only has a very small
            effect on the environment.
        callbacks (list of `keras.callbacks.Callback` or
            `rl.callbacks.Callback` instances): List of callbacks to apply
            during training.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled '
                + 'yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(
                    action_repetition))

        self.training = True
        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks.append(history)
        callbacks = CallbackList(callbacks)

        actors = [self.learner.create_actor(i) for i in range(self.num_actors)]
        envs = [env_factory(i) for i in range(self.num_actors)]

        self.step = 0
        callbacks.on_train_begin()
        while self.step < nb_steps:
            max_horizon = nb_steps - self.step
            trajectories = [actor.build_trajectory(env, max_horizon, callbacks)
                            for actor, env in zip(actors, envs)]

            # TODO how to increase step?
            # Currently, the length of the longest trajectory is added
            # Which means that in overall, all the actors will do (much) fewer
            # steps
            self.step += len(max(trajectories, key = len))

            learner_history = self.learner.backward(trajectories)
            step_logs = {
                'actor': None,
                'learner_history': learner_history,
                }
            # TODO finalise and document the API for this
            callbacks.on_step_end(self.step, step_logs)

        # No support for keyboard interrupt yet.
        callbacks.on_train_end(logs = {'did_abort': False})
        return history

    def test(self, env_factory, nb_episodes, action_repetition = 1,
             callbacks = None, nb_max_episode_steps = None):
        """
        Tests the agent on the given environment.

        Only one actor is tested on one environment instance.

        # Arguments:
        env_factory (`lambda integer: env`): A factory function that gives an
            new environment instance. Its parameter is the index of the
            instance; this will be 0.
        nb_episodes (integer): Number of training episodes to perform.
        action_repetition (integer): Number of times the agent repeats the same
            action without observing the environment again. Setting this to a
            value > 1 can be useful if a single action only has a very small
            effect on the environment.
        callbacks (list of `keras.callbacks.Callback` or
            `rl.callbacks.Callback` instances): List of callbacks to apply
            during training.
        nb_max_episode_steps (integer): Number of steps per episode that the
            agent performs before automatically resetting the environment. Set
            to `None` if each episode should run (potentially indefinitely)
            until the environment signals a terminal state.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled '
                + 'yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(
                    action_repetition))

        self.training = False
        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks.append(history)
        callbacks = CallbackList(callbacks)

        actor = self.learner.create_actor(0)
        env = env_factory(0)

        self.step = 0

        # TODO remove the double counting of episodes (here and in the Actor)
        callbacks.on_train_begin()
        for _ in range(nb_episodes):
            # This will reset the environment in the next `build_trajectory`
            # call.
            actor.reset()
            if nb_max_episode_steps is None:
                # Collect a relatively long trajectory, then continue if not
                # done yet.
                done = False
                episode_step = 0
                while not done:
                    done, trajectory = actor.build_trajectory(env, 1000,
                                                              callbacks)
                    episode_step += len(trajectory) - 1
            else:
                done, trajectory = actor.build_trajectory(
                    env, nb_max_episode_steps, callbacks)
                episode_step = len(trajectory) - 1
                episode_reward = sum(r for s,a,r in trajectory[:-1])

            self.step += episode_step

            # TODO do this for fit also
            if not done:
                # Logs of the final (not finished) episode
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_episode_steps': episode_step,
                    'nb_steps': self.step,
                }
                callbacks.on_episode_end(actor.episode, episode_logs)

        callbacks.on_train_end()

        return history
