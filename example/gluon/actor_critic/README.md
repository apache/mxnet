<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Actor Critic Model

This example shows an actor critic model that consists of a critic that measures how good an action taken is and an actor that controls the agent's behavior. 
In our example actor and critic use the same model:

```
class Policy(gluon.Block):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(16, in_units=4, activation='relu')
            self.action_pred = nn.Dense(2, in_units=16)
            self.value_pred = nn.Dense(1, in_units=16)

    def forward(self, x):
        x = self.dense(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return F.softmax(probs), values
```
The example uses [Gym](https://gym.openai.com/docs/), which is a toolkit for developing and comparing reinforcement learning algorithms. The model is running an instance of [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) that simulates a pole that is attached by an un-actuated joint to a cart, which moves along a frictionless track. The goal is to prevent it from falling over. 


The example provides the following commandline options:
```
MXNet actor-critic example

optional arguments:
  -h, --help        show this help message and exit
  --gamma G         discount factor (default: 0.99)
  --seed N          random seed (default: 1)
  --render          render the environment
  --log-interval N  interval between training status logs (default: 10)

```

To run the model execute, type 
```
python actor_critic.py --render
```

You will get an output like the following:
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/actor_critic/actor_critic.gif)

