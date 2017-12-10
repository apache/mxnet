# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Trains an `Agent` using trajectories from multiple environments."""

import argparse
from itertools import chain
import time
import gym
import numpy as np
import mxnet as mx
from config import Config
from envs import Atari8080Preprocessor, IdentityPreprocessor
from model import Agent


def train_episode(agent, envs, preprocessors, t_max, render):
    """Complete an episode's worth of training for each environment."""
    num_envs = len(envs)

    # Buffers to hold trajectories, e.g. `env_xs[i]` will hold the observations
    # for environment `i`.
    env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
    env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
    episode_rs = np.zeros(num_envs, dtype=np.float)

    for p in preprocessors:
        p.reset()

    observations = [p.preprocess(e.reset())
                    for p, e in zip(preprocessors, envs)]

    done = np.array([False for _ in range(num_envs)])
    all_done = False
    t = 1

    while not all_done:
        if render:
            envs[0].render()

        # NOTE(reed): Reshape to set the data shape.
        agent.model.reshape([('data', (num_envs, preprocessors[0].obs_size))])
        step_xs = np.vstack([o.ravel() for o in observations])

        # Get actions and values for all environments in a single forward pass.
        step_xs_nd = mx.nd.array(step_xs, ctx=agent.ctx)
        data_batch = mx.io.DataBatch(data=[step_xs_nd], label=None)
        agent.model.forward(data_batch, is_train=False)
        _, step_vs, _, step_ps = agent.model.get_outputs()

        step_ps = step_ps.asnumpy()
        step_vs = step_vs.asnumpy()
        step_as = agent.act(step_ps)

        # Step each environment whose episode has not completed.
        for i, env in enumerate(envs):
            if not done[i]:
                obs, r, done[i], _ = env.step(step_as[i])

                # Record the observation, action, value, and reward in the
                # buffers.
                env_xs[i].append(step_xs[i].ravel())
                env_as[i].append(step_as[i])
                env_vs[i].append(step_vs[i][0])
                env_rs[i].append(r)
                episode_rs[i] += r

                # Add 0 as the state value when done.
                if done[i]:
                    env_vs[i].append(0.0)
                else:
                    observations[i] = preprocessors[i].preprocess(obs)

        # Perform an update every `t_max` steps.
        if t == t_max:
            # If the episode has not finished, add current state's value. This
            # will be used to 'bootstrap' the final return (see Algorithm S3
            # in A3C paper).
            step_xs = np.vstack([o.ravel() for o in observations])
            step_xs_nd = mx.nd.array(step_xs, ctx=agent.ctx)
            data_batch = mx.io.DataBatch(data=[step_xs_nd], label=None)
            agent.model.forward(data_batch, is_train=False)
            _, extra_vs, _, _ = agent.model.get_outputs()
            extra_vs = extra_vs.asnumpy()
            for i in range(num_envs):
                if not done[i]:
                    env_vs[i].append(extra_vs[i][0])

            # Perform update and clear buffers.
            env_xs = np.vstack(list(chain.from_iterable(env_xs)))
            agent.train_step(env_xs, env_as, env_rs, env_vs)
            env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
            env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
            t = 0

        all_done = np.all(done)
        t += 1

    return episode_rs


def _2d_list(n):
    return [[] for _ in range(n)]


def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--t-max', type=int, default=50)
    parser.add_argument('--env-type', default='PongDeterministic-v4')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')

    # Parse arguments and setup configuration `config`
    args = parser.parse_args()
    config = Config(args)
    print('args=%s' % args)
    print('config=%s' % config.__dict__)
    np.random.seed(args.seed)

    # Create and seed the environments
    envs = [gym.make(args.env_type) for _ in range(args.num_envs)]
    if args.env_type == 'CartPole-v0':
        preprocessors = [
            IdentityPreprocessor(np.prod(envs[0].observation_space.shape))
            for _ in range(args.num_envs)]
    else:
        preprocessors = [Atari8080Preprocessor() for _ in range(args.num_envs)]
    for i, env in enumerate(envs):
        env.seed(i+args.seed)

    agent = Agent(preprocessors[0].obs_size, envs[0].action_space.n,
                  config=config)

    # Train
    running_reward = None
    start = time.time()
    for i in range(args.num_episodes):
        tic = time.time()
        episode_rs = train_episode(
            agent, envs, preprocessors, t_max=args.t_max, render=args.render)

        for er in episode_rs:
            running_reward = er if running_reward is None else (
                0.99 * running_reward + 0.01 * er)

        if i % args.print_every == 0:
            print('Batch %d complete (%.2fs) (%.1fs elapsed) (episode %d), '
                  'batch avg. reward: %.2f, running reward: %.3f' %
                  (i, time.time() - tic, time.time() - start,
                   (i + 1) * args.num_envs, np.mean(episode_rs),
                   running_reward))

        if args.save_every > 0:
            if i % args.save_every == 0:
                save_params(args.save_pre, agent.model, i)
