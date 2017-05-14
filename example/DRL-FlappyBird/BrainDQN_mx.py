# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import mxnet as mx 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100
ctx=mx.cpu()

def dataPrep(data):
    if data.shape[2]>1:
        mean = np.array([128, 128, 128,128])
        reshaped_mean = mean.reshape(1, 1, 4)
    else:
        mean=np.array([128])
        reshaped_mean = mean.reshape(1, 1, 1)
    img = np.array(data, dtype=np.float32)
    data = data - reshaped_mean
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    data = np.expand_dims(data, axis=0)
    return data


class BrainDQN:
    def __init__(self,actions,param_file=None):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.target = self.createQNetwork(isTrain=False)

        self.Qnet = self.createQNetwork()
        if param_file!=None:
            self.Qnet.load_params(param_file)
        self.copyTargetQNetwork()
        # saving and loading networks

    def sym(self,predict=False):
        data = mx.sym.Variable('data')
        yInput = mx.sym.Variable('yInput')
        actionInput = mx.sym.Variable('actionInput')
        conv1 = mx.sym.Convolution(data=data,kernel=(8,8),stride=(4,4),pad=(2,2),num_filter=32,name='conv1')
        relu1 = mx.sym.Activation(data=conv1,act_type='relu',name='relu1')
        pool1 = mx.sym.Pooling(data=relu1,kernel=(2,2),stride=(2,2),pool_type='max',name='pool1')
        conv2 = mx.sym.Convolution(data=pool1,kernel=(4,4),stride=(2,2),pad=(1,1),num_filter=64,name='conv2')
        relu2 = mx.sym.Activation(data=conv2,act_type='relu',name='relu2')
        conv3 = mx.sym.Convolution(data=relu2,kernel=(3,3),stride=(1,1),pad=(1,1),num_filter=64,name='conv3')
        relu3 = mx.sym.Activation(data=conv3,act_type='relu',name='relu3')
        flat  = mx.sym.Flatten(data=relu3,NameError='flat')
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=512,name='fc1')
        relu4 = mx.sym.Activation(data=fc1,act_type='relu',name='relu4')
        Qvalue = mx.sym.FullyConnected(data=relu4, num_hidden=self.actions,name='qvalue')
        temp=Qvalue*actionInput
        coeff=mx.sym.sum(temp,axis=1,name='temp1')
        output = (coeff - yInput)**2
        loss=mx.sym.MakeLoss(output)

        if predict:
            return Qvalue
        else:
            return loss

    def createQNetwork(self,bef_args=None,isTrain=True):
        if isTrain:
            modQ = mx.mod.Module(symbol=self.sym(), data_names=('data','actionInput'), label_names=('yInput',), context=ctx)
            batch=BATCH_SIZE
            modQ.bind(data_shapes=[('data',(batch,4,80,80)),('actionInput',(batch,self.actions))],
                        label_shapes=[('yInput',(batch,))],
                        for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)
            modQ.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': 0.0002,
                    'wd': 0.,
                    'beta1': 0.5,
            })
        else:
            modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=ctx)
            batch=1
            modQ.bind(data_shapes=[('data',(batch,4,80,80))],
                        for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)

        return modQ
             
    def copyTargetQNetwork(self):
        arg_params,aux_params=self.Qnet.get_params()
        #arg={}
        #for k,v in arg_params.iteritems():
        #    arg[k]=arg_params[k].asnumpy()

        self.target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)

        #args,auxs=self.target.get_params()
        #arg1={}
        #for k,v in args.iteritems():
        #    arg1[k]=args[k].asnumpy()
        print 'time to copy'

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = np.squeeze([data[0] for data in minibatch])
        action_batch =  np.squeeze([data[1] for data in minibatch])
        reward_batch =  np.squeeze([data[2] for data in minibatch])
        nextState_batch =  [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = np.zeros((BATCH_SIZE,))
        Qvalue=[]
        for i in range(BATCH_SIZE):
            self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch[i],ctx)],[]))
            Qvalue.append(self.target.get_outputs()[0].asnumpy())
        Qvalue_batch=np.squeeze(Qvalue)
        terminal=np.squeeze([data[4] for data in minibatch])
        y_batch[:]=reward_batch
        if (terminal==False).shape[0]>0:
            y_batch[terminal==False]+= (GAMMA * np.max(Qvalue_batch,axis=1))[terminal==False]

        self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,ctx),mx.nd.array(action_batch,ctx)],[mx.nd.array(y_batch,ctx)]),is_train=True)
        self.Qnet.backward()
        self.Qnet.update()

        # save network every 1000 iteration
        if self.timeStep % 100 == 0:
            self.Qnet.save_params('saved_networks/network-dqn_mx%04d.params'%(self.timeStep))

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()


    def setInitState(self,observation):
        temp=dataPrep(np.stack((observation, observation, observation, observation), axis = 2))
        self.currentState = temp
    
    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)

        newState = np.append(self.currentState[:,1:,:,:],dataPrep(nextObservation),axis = 1)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print "TIMESTEP", self.timeStep, "/ STATE", state, \
        "/ EPSILON", self.epsilon

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        self.target.forward(mx.io.DataBatch([mx.nd.array(self.currentState,ctx)],[]))
        QValue=np.squeeze(self.target.get_outputs()[0].asnumpy())
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action