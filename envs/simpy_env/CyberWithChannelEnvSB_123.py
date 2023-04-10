# -*- coding: utf-8 -*-
"""
Created on Fri July 1 15:24:05 2022

@author: abhijeetsahu

This environment is created for episodic interaction of the Simpy based COmmunication simulator for training RL agent, similar to an Open AI Gym environment
with implementation of the channel model
"""
from distutils.log import debug
from tkinter import CURRENT
from xml.dom.minicompat import NodeList
import gym
import random
import functools
from multiprocessing import Event
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from envs.simpy_env.SimComponentsDynamic import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Router, Firewall,Packet
from envs.simpy_env.simtools import SimMan,Notifier
from abc import ABC, abstractmethod
import simpy
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, List, Tuple
from queue import Queue


def constArrival():
    #return 100000000    # time interval
    return 13.0

def constArrival2():
    #return 100000000    # time interval
    return 40.0

def constSize():
    return 25.0  # bytes

def constSize2():
    return 100.0  # bytes


adist = functools.partial(random.expovariate, 0.5)
sdist = functools.partial(random.expovariate, 0.04)  # mean size 100 bytes

#adist2 = functools.partial(random.expovariate, 0.25)
#adist2 = functools.partial(random.randint, 45,55)
adist2 = functools.partial(random.randint, 12,15)
sdist2 = functools.partial(random.expovariate, 0.0005)  # mean size 100 bytes


samp_dist = functools.partial(random.expovariate, 1.0)
port_rate = 1000.0

class Message:
    """
    A class used for the exchange of arbitrary messages between components.
    A :class:`Message` can be used to simulate both asynchronous and synchronous function
    calls.
    Attributes:
        type(Enum): An enumeration object that defines the message type
        args(Dict[str, Any]): A dictionary containing the message's arguments
        eProcessed(Event): A SimPy event that is triggered when
            :meth:`setProcessed` is called. This is useful for simulating
            synchronous function calls and also allows for return values (an
            example is provided in :meth:`setProcessed`).
    """

    def __init__(self, type: Enum, args: Dict[str, Any] = None):
        self.type = type
        self.args = args
        self.eProcessed = Event(SimMan.env)

    def setProcessed(self, returnValue: Any = None):
        """
        Makes the :attr:`eProcessed` event succeed.
        Args:
            returnValue: If specified, will be used as the `value` of the
                :attr:`eProcessed` event.
        Examples:
            If `returnValue` is specified, SimPy processes can use Signals for
            simulating synchronous function calls with return values like this:
            ::
                signal = Signal(myType, {"key", value})
                gate.output.send(signal)
                value = yield signal.eProcessed
                # value now contains the returnValue that setProcessed() was called with
        """
        self.eProcessed.succeed(returnValue)
    
    def __repr__(self):
        return "Message(type: '{}', args: {})".format(self.type.name, self.args)

class StackMessageTypes(Enum):
    """
    An enumeration of control message types to be used for the exchange of
    `Message` objects between network stack layers.
    """
    RECEIVE = 0
    SEND = 1
    ASSIGN = 2

class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_RECEIVER_DEGREE = 5 

    ASSIGNMENT_DURATION_FACTOR = 1000

    def __init__(self,  deviceCount: int):
        """
        Args:
            deviceCount: The number of devices to be included in the
                environment's action space
        """
        self.deviceCount = deviceCount
        self.action_space = spaces.Dict({
            "device": spaces.Discrete(deviceCount),
            "next_hop": spaces.Discrete(self.MAX_RECEIVER_DEGREE)
        })

        self.seed()

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator and returns
        it in a single-item list.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        """
        Renders the environment to stdout.
        """

class Interpreter(ABC):
    """
    An :class:`Interpreter` is an instance that observes the system's behavior
    by sniffing the packets received by the receiver and infers
    observations and rewards.
    This class serves as an abstract base class for all :class:`Interpreter`
    implementations.
    When implementing an interpreter, the following three methods have to be
    overridden:
        * :meth:`getReward`
        * :meth:`getObservation`
    The following methods provide default implementations that you might also
    want to override depending on your use case:
        * :meth:`reset`
        * :meth:`getDone`
        * :meth:`getInfo`
    """

    @abstractmethod
    def getReward(self) -> float:
        """
        Returns a reward that depends on the last channel assignment.
        """

    @abstractmethod
    def getObservation(self) -> Any:
        """
        Returns an observation of the system's state.
        """
    
    def getDone(self) -> bool:
        """
        Returns whether an episode has ended.
        Note:
            Reinforcement learning problems do not have to be split into
            episodes. In this case, you do not have to override the default
            implementation as it always returns ``False``.
        """
        return False

    def getInfo(self) -> Dict:
        """
        Returns a :class:`dict` providing additional information on the
        environment's state that may be useful for debugging but is not allowed
        to be used by a learning agent.
        """
        return {}

    def getFeedback(self) -> Tuple[Any, float, bool, Dict]:
        """
        You may want to call this at the end of a frequency band assignment to get
        feedback for your learning agent. The return values are ordered like
        they need to be returned by the :meth:`step` method of a gym
        environment.
        Returns:
            A 4-tuple with the results of :meth:`getObservation`,
            :meth:`getReward`, :meth:`getDone`, and :meth:`getInfo`
        """
        return self.getObservation(), self.getReward(), self.getDone(), self.getInfo()
    
    def reset(self):
        """
        This method is invoked when the environment is reset – override it with
        your initialization tasks if you feel like it.
        """


class CyberEnv(BaseEnv):

    class SenderDevice(PacketGenerator):
        """
        """
        
        def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0,debug=True,packet_size=None):
            super(CyberEnv.SenderDevice, self).__init__(env, id,  adist, sdist)
            self.initial_delay=initial_delay
            self.out_channel = None
            self.debug=debug
            self.packet_size = packet_size
            SimMan.process(self.senderProcess())
        
        def senderProcess(self):
            yield SimMan.timeout(self.initial_delay)
            while SimMan.now < self.finish:
                # wait for next transmission
                yield SimMan.timeout(self.adist())
                self.packets_sent += 1

                p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
                if self.packet_size is not None:
                    p.size = float(self.packet_size)

                #self.out.put(p)
                self.out_channel.put(p)

                if self.debug:
                    print('gen '+str(self.env.now)+' '+str(p))

        def sendCommand(self):
            yield SimMan.timeout(self.adist())
            self.packets_sent +=1
            p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            if self.debug:
                print(str(self.id) + ' : Sending control command')
            self.out.put(p)

    class ForwarderDevice(SwitchPort):
        def __init__(self, env,id, rate, qlimit=None, limit_bytes=True, debug=False):
            super(CyberEnv.ForwarderDevice, self).__init__(env,id, rate, qlimit, limit_bytes, debug)
            self.out_channel = []
            self.selected_Channel_Index = 0
            self.queue_update_freq=25
            SimMan.process(self.forwarderProcess())
            SimMan.process(self.update_router_queue())
            #self.notifier = Notifier('Updates', self)
        
        def forwarderProcess(self):
            while True:
                msg = (yield self.store.get())
                self.busy = 1
                self.byte_size -= msg.size
                # take some time for the router to process the packet
                yield SimMan.timeout(msg.size*8.0/(self.rate))
                try:
                    # here the out should be the channel instead of the router
                    self.out_channel[self.selected_Channel_Index].put(msg)
                except:
                    print(self.id)
                    #print('*************************************************************************************************************')
                #self.out[0].put(msg)
                self.busy = 0
                if self.debug:
                    print('No of outchannels {0} their ids {1}'.format(len(self.out_channel),[x.cid for x in self.out_channel]))
                    print('At '+str(self.env.now)+', '+str(self.id)+':'+str(self.env.now)+' : ' +str(msg))
                    print(str(self.env.now)+' : Selected Channel of Router {0} is {1}'.format(self.id, self.selected_Channel_Index))

        def update_router_queue(self):
            while True:
                if self.debug:
                    print('At {0}, {1} percent filled : {2}'.format(self.env.now, self.id,self.temp_byte_size/self.qlimit))
                self.temp_byte_size = 0
                yield SimMan.timeout(self.queue_update_freq)

        # this function will change the receiver if it founds one receiver to be busy
        def change_receiver(self, new_receiver):
            #self.out = [new_receiver]
            # the out channel depends on the receiving router selected
            #self.out_channel = [new_receiver]
            self.selected_Channel_Index = new_receiver
            #print(str(self.env.now)+' : '+str(self.id) + ' : Changing Route to ' + str(self.out[self.selected_Channel_Index].id))
            if self.debug:
                print(str(self.env.now)+' : '+str(self.id) + ' : Changing Route to ' + str(self.out[self.selected_Channel_Index].id))
            yield SimMan.timeout(1)

    class ReceiverDevice(Interpreter):
        def __init__(self, simpyenv,id,gymenv, rec_arrivals=False, absolute_arrivals=False, rec_waits=True, debug=True, selector=None):
            self.store = simpy.Store(simpyenv)
            self.env = simpyenv
            self.gymenv = gymenv
            self.id = id
            self.rec_waits = rec_waits
            self.rec_arrivals = rec_arrivals
            self.absolute_arrivals = absolute_arrivals
            self.waits = []
            self.arrivals = []
            self.debug = debug
            self.packets_rec = 0
            self.packets_record ={}
            self.bytes_rec = 0
            self.selector = selector
            self.last_arrival = 0.0
            self.reset()
            

        def put(self, pkt):
            if not self.selector or self.selector(pkt):
                now = self.env.now
                #if self.rec_waits and pkt.src in ['PG11','PG12','PG13','PG14','PG15','PG16','PG17']:
                if self.gymenv.small_Case:
                    if self.rec_waits and pkt.src in ['PG1']:# small case
                        self.waits.append(self.env.now - pkt.time)
                else:
                    if self.rec_waits and pkt.src in ['PG11','PG12','PG13','PG14','PG15','PG16','PG17']:
                        self.waits.append(self.env.now - pkt.time)
                if self.rec_arrivals:
                    if self.absolute_arrivals:
                        self.arrivals.append(now)
                    else:
                        self.arrivals.append(now - self.last_arrival)
                    self.last_arrival = now
                #print('Packet Sorce '+str(pkt.src))
                #if pkt.src in ['PG11','PG12','PG13','PG14','PG15','PG16','PG17']:
                if self.gymenv.small_Case:
                    if pkt.src in ['PG1']:
                        self.packets_rec += 1
                        if pkt.src in self.packets_record.keys():
                            self.packets_record[pkt.src]+=1
                        else:
                            self.packets_record[pkt.src]=1
                else:
                    if pkt.src in ['PG11','PG12','PG13','PG14','PG15','PG16','PG17']:
                        self.packets_rec += 1
                        if pkt.src in self.packets_record.keys():
                            self.packets_record[pkt.src]+=1
                        else:
                            self.packets_record[pkt.src]=1
                self.bytes_rec += pkt.size
                if self.debug:
                    print('At '+str(self.env.now)+', '+str(self.id)+':'+str(self.env.now)+' '+str(pkt))
        
        def reset(self):
            self.receivedPackets = [0 for _ in range(len(self.gymenv.senders))]
            self._done = False

        def getReward(self):
            """
            Will depend on the percentage of successfully packet received (WE WILL FURTHER incorprorate other resilience metric)
            """
            try:
                #print('Packets Rec '+str(self.packets_rec))
                '''
                bound = 70
                if self.gymenv.small_Case:
                    bound = 10
                if self.packets_rec >= bound/2 and self.packets_rec < bound: #small case
                    reward = -1
                elif self.packets_rec <= bound/2:
                    reward = -2
                else:
                    reward = 2
                '''
                gthan2 = dict((k, v) for k, v in self.packets_record.items() if v >= 2)
                if len(gthan2) == 7:
                    reward = 5
                elif len(gthan2) == 6:
                    reward = 0
                else:
                    reward = -2
                
                #reward = -(1/self.packets_rec)-(self.gymenv.routers[1].packets_drop)
                #reward = -((1/self.packets_rec) + float(self.env.routers[1].packets_drop)/self.env.routers[1].packets_rec)
                #avg_wait = sum(self.waits)/len(self.waits)
                #reward = float(reward / sum(self.waits))
                #print('Reward '+str(reward))
                return float(reward)
                #return float(1/avg_wait)
            except:
                return 0.0

        def getObservation(self):
            # instead of random value lets  get some value
            drop_rate_val = []
            for router in self.gymenv.routers:
                drop_rate_val.append(router.packets_drop)

            # we add the channel utilization rate to this also and construct the weighted graph for shortest path
            channel_urs = []
            for channel in self.gymenv.channels:
                channel_urs.append(channel.utilization_rate)

            if self.gymenv.channelModel:
                drop_rate_val = channel_urs
            return np.array(drop_rate_val)
        
        def getDone(self):
            #print(self.packets_record)
            
            gthan2 = dict((k, v) for k, v in self.packets_record.items() if v >= 2)
            if len(gthan2) == 7:
                self._done = True
            """
            if self.packets_rec > 70:
                self._done = True
            
            #small case
            if self.gymenv.small_Case:
                if self.packets_rec > 10:
                    self._done = True
            else:
                if self.packets_rec > 70:
                    self._done = True
            """
            """ 
            if self.packets_rec > 100:
                    self._done = True
                    #if self.debug:
                    print(self.packets_record)
                 #print ('time of done '+str(self.env.now)) 
            total_loss = 0
            ctr = 0
            for rtrs in self.gymenv.routers:
                if rtrs.packets_rec > 0:
                    total_loss += float(rtrs.packets_drop)/rtrs.packets_rec
                    ctr+=1
            avg_lossrate = total_loss/ctr
            print ('Avg Loss Rate : {0}'.format(avg_lossrate))
            if avg_lossrate < 0.09:
                self._done = True """
            return self._done
        
        def getInfo(self):
            # DQN in keras-rl crashes when the values are iterable, thus the
            # string below
            return {"Last arrived packet": str(self.last_arrival)}

    # the overall delay depends on multiple factors : transmission delay + propagation delay + queueing delay + processing delay
    # the transmission delay depends on the channel capacity
    # the propagation delay : distance/speed of transmitting medium such as light or sound
    # the queueing delay is the wait time and processing at every switch and router
    # while the processing delay is all the time consumed in encapsulation and decapsulation of headers + fragmentation and reassembly etc etc

    # In our simulation, we implement dynamic update on channel_capacity that changes the channel capacity which changes the transmission delay and keep 
    # the propagation delay fixed
    class Channel():
        def __init__(self,env,id, src,dest, bw=2000, delay=1, limit_bytes=True,debug = False, snr = 10):
            self.store = simpy.Store(env)
            self.cid = id
            self.packets_rec = 0
            self.packets_drop = 0
            self.env = env # simpy env
            self.src = src # source node
            self.dest = dest # destination node
            self.bw = bw # channel bandwidth
            self.byte_size = 0  # Number of data size already in the channel
            self.delay = delay # channel delay (this should be propagation delay)
            self.temp_byte_size = 0
            self.limit_bytes = limit_bytes
            self.debug = debug
            #self.channel_capacity = self.bw * math.log10(1 + snr) # shannon's channel capacity formula for noisy channel
            # for noiseless channel
            # the utilization rate is usually computed in a time interval how much bytes are served
            self.utilization_rate = 0
            self.channel_capacity = self.bw
            self.ur_update_freq=25
            SimMan.process(self.run())
            SimMan.process(self.update_ur())

        def run(self):
            while True:
                msg = (yield self.store.get())
                # this first expression is transmission delay and second the propagation
                try:
                    latency =  msg.size*8.0/self.channel_capacity  + self.delay
                except:
                    latency =  msg.size*8.0/(self.channel_capacity+100)  + self.delay
                if latency < 0:
                    latency = 0
                yield SimMan.timeout(latency) 
                """ try:
                    yield SimMan.timeout(latency) 
                except:
                    print(self.channel_capacity) """
                self.dest.put(msg)
                if self.debug:
                    print(msg)

        def put(self, pkt):
            self.packets_rec += 1
            tmp_byte_count = self.byte_size + pkt.size
            self.channel_capacity = self.bw * (1 - self.utilization_rate)
            #print('{0} I am called by router'.format(self.cid))
            if self.debug:
                print('{0}: testing channel capacity {1} and tmp_byte_size {2}'.format(self.env.now, self.channel_capacity, self.temp_byte_size))
                print('{0}: utilization rate: {1}'.format(self.cid, self.utilization_rate))
            if self.channel_capacity is None:
                self.byte_size = tmp_byte_count
                self.temp_byte_size = self.byte_size
                return self.store.put(pkt)
            if self.limit_bytes and self.temp_byte_size >= self.channel_capacity:
                self.packets_drop += 1
                return
            elif not self.limit_bytes and len(self.store.items) >= self.channel_capacity-1:
                self.packets_drop += 1
            else:
                self.byte_size = tmp_byte_count
                self.temp_byte_size+= self.byte_size
                return self.store.put(pkt)

        # schedule every 10 sec to update the Utilization Rate
        def update_ur(self):
            while True:
                self.utilization_rate = self.temp_byte_size/(self.ur_update_freq*self.bw)
                if self.debug:
                    print('At {0}, {1} update  utilization val : {2}'.format(self.env.now, self.cid,str(self.utilization_rate)))
                self.temp_byte_size = 0
                yield SimMan.timeout(self.ur_update_freq)

    class WiredChannel(Channel):
        def __init__(self,env, src,dest):
            super(CyberEnv.WiredChannel,self).__init__(env,src,dest)
            SimMan.process(self.run())

        def run(self):
            NotImplementedError

    class WirelessChannel(Channel):
        def __init__(self,env, src,dest):
            super(CyberEnv.WirelessChannel,self).__init__(env,src,dest)
            SimMan.process(self.run())

        def run(self):
            NotImplementedError

    ###### Continuing with the constructor of the Cyber Environment#################################

    def __init__(self, provided_graph= None,channelModel=True, envDebug=True, R2_qlimit = 100, ch_bw=1000, with_threat= False, comp_zones = None, dev_compromised = 1):
        super(CyberEnv, self).__init__(deviceCount=4)
        self.channelModel = channelModel
        self.envDebug = envDebug
        self.R2_qlimit = R2_qlimit
        self.ch_bw = ch_bw
        self.channel_map ={}
        self.with_threat = with_threat
        self.comp_zones = comp_zones
        self.small_Case = False
        if provided_graph is None:
            self.small_Case = True
            # Here we monitor the number of lost packets at all the router nodes (which is kind of the device count)
            self.G = nx.Graph()
            self.reinitialize_network()

            if channelModel is True:
                self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(len(self.channels),), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(self.deviceCount,), dtype=np.float32)
            
            self.action_space = spaces.MultiDiscrete([self.deviceCount, 2])
            #self.action_space = spaces.MultiDiscrete([1, 2])

        else:
            #self.deviceCount = 16
            self.deviceCount = 18
            self.device_compromised = dev_compromised
            self.G = provided_graph
            self.reinitialize_complex_network(self.G)
            """ self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(self.deviceCount + len(self.channels),), dtype=np.float32)
            self.action_space = spaces.Dict({
            "device": spaces.Discrete(self.deviceCount),
            "next_hop": spaces.Discrete(self.deviceCount)
            }) """
            if channelModel is True:
                self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(len(self.channels),), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(self.deviceCount,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([self.deviceCount, 5])


    def reinitialize_network(self):
        SimMan.init()
        self.nodes = []
        self.edges = []
        self.senders: List[self.SenderDevice] = [
            #CyberEnv.SenderDevice(SimMan.env, 'PG1', adist, sdist ),
            #CyberEnv.SenderDevice(SimMan.env, 'PG1',  constArrival, constSize),
            #CyberEnv.SenderDevice(SimMan.env, 'PG2', adist2, sdist2)

            CyberEnv.SenderDevice(SimMan.env, 'PG1', constArrival, sdist ,debug=False),
            #CyberEnv.SenderDevice(SimMan.env, 'PG1',  constArrival, constSize),
            CyberEnv.SenderDevice(SimMan.env, 'PG2', constArrival2, constSize,debug=False)
        ]
        self.nodes.extend(self.senders)

        # initialize all the forwarding devices
        self.routers: List[self.ForwarderDevice] = [
            CyberEnv.ForwarderDevice(SimMan.env, 'R1',rate=200.0, qlimit=200,debug=self.envDebug),
            CyberEnv.ForwarderDevice(SimMan.env,'R2',rate=200.0, qlimit=self.R2_qlimit,debug=self.envDebug),
            CyberEnv.ForwarderDevice(SimMan.env,'R3',rate=200.0, qlimit=200,debug=self.envDebug),
            CyberEnv.ForwarderDevice(SimMan.env,'R4',rate=200.0, qlimit=200,debug=self.envDebug)
        ]
        
        self.nodes.extend(self.routers)
        
        self.interpreter = self.ReceiverDevice(SimMan.env, 'PS',self, debug=self.envDebug)
        self.nodes.extend([self.interpreter])

        self.channels : List[self.channels] = [
            CyberEnv.Channel(SimMan.env, 'CG1',src = self.senders[0],dest=self.routers[0],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'CG2',src = self.senders[1],dest=self.routers[1],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'C12',src = self.routers[0],dest=self.routers[1],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'C13',src = self.routers[0],dest=self.routers[2],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'C24',src = self.routers[1],dest=self.routers[3],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'C34',src = self.routers[2],dest=self.routers[3],bw=self.ch_bw),
            CyberEnv.Channel(SimMan.env, 'CS',src = self.routers[3],dest=self.interpreter,bw=self.ch_bw)
        ]
        self.edges.extend(self.channels)


        for node in self.nodes:
            self.G.add_node(node.id)

        # create the network, i.e. connect the edges
        self.senders[0].out = self.routers[0]
        self.senders[1].out = self.routers[1]
        self.routers[0].out.append(self.routers[1])
        self.routers[0].out.append(self.routers[2])
        self.routers[1].out.append(self.routers[3])
        self.routers[2].out.append(self.routers[3])
        self.routers[3].out.append(self.interpreter)

        # initializa the channels
        self.senders[0].out_channel = self.channels[0]
        self.senders[1].out_channel = self.channels[1]
        self.routers[0].out_channel.append(self.channels[2])
        self.routers[0].out_channel.append(self.channels[3])
        self.routers[1].out_channel.append(self.channels[4])
        self.routers[2].out_channel.append(self.channels[5])
        self.routers[3].out_channel.append(self.channels[6])

        edges = [(self.senders[0],self.routers[0]), (self.senders[1],self.routers[1]), (self.routers[0],self.routers[1]), 
        (self.routers[0],self.routers[2]), (self.routers[1],self.routers[3]), (self.routers[2],self.routers[3]), (self.routers[3],self.interpreter)]

        #edges = [(self.senders[0],self.routers[0]), (self.routers[0],self.routers[1]), 
        #(self.routers[0],self.routers[2]), (self.routers[1],self.routers[3]), (self.routers[2],self.routers[3]), (self.routers[3],self.interpreter)]


        for edge in edges:
            self.G.add_edge(edge[0].id,edge[1].id, weight = 0.0)


    # lets say we are given a graph we need to construct the simpy network for a larger system
    # G.add_node(n, nodetype='sender', 'router' or 'sink')
    def reinitialize_complex_network(self, G):
        SimMan.init()
        self.nodes = []
        self.edges = []
        self.senders = []
        self.routers = []
        self.channels = []
        self.interpreter = None
        sender_ix = 0
        for key,node in G.nodes(data=True):
            if node['nodetype'] == 'sender':

                if self.comp_zones is None:
                    g_node =  CyberEnv.SenderDevice(SimMan.env, key, constArrival, sdist, initial_delay=random.randint(10,70),debug=self.envDebug )
                else:
                    components_in_zone = sum(x == sender_ix+1 for x in self.comp_zones.values())
                    g_node =  CyberEnv.SenderDevice(SimMan.env, key, constArrival, constSize, initial_delay=random.randint(10,70),debug=self.envDebug,packet_size=components_in_zone)
                self.senders.append(g_node)
                sender_ix+=1
            elif node['nodetype'] == 'router':
                r_node = CyberEnv.ForwarderDevice(SimMan.env, 'R'+str(key),rate=400.0, qlimit=self.R2_qlimit,debug=self.envDebug)
                self.routers.append(r_node)
            elif node['nodetype'] == 'sink':
                self.interpreter = self.ReceiverDevice(SimMan.env, key,self, debug=self.envDebug)
        self.nodes.extend(self.senders)
        self.nodes.extend(self.routers)
        self.nodes.extend([self.interpreter])

        # add the channels
        f_ctr = 0
        # for each sender first connect all the channels to their respective routers
        for ix,s_node in enumerate(self.senders):
            src_node = [x for x in self.senders if x.id == s_node.id][0]
            dest_node = [x for x in self.routers if x.id == 'R'+s_node.id[2:]][0]
            self.channels.append(CyberEnv.Channel(SimMan.env, 'CG'+str(ix+1), src = src_node, dest=dest_node,bw=self.ch_bw,debug=self.envDebug))
            self.senders[ix].out_channel = self.channels[ix]
            self.senders[ix].out = dest_node
            self.channel_map[f_ctr] = [src_node.id, dest_node.id]
            f_ctr+=1

        # add the channels among the routers
        for edge in G.edges(data=True):
            if 'PS' not in str(edge[1]) and 'PG' not in str(edge[1]):
                src = [x for x in self.nodes if x.id == 'R'+str(edge[1])][0]
                dest = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]
                if dest not in src.out:
                    src.out.append(dest)
                ch = CyberEnv.Channel(SimMan.env, 'C_'+src.id[1:]+'_'+dest.id[1:], src = src, dest=dest,bw=self.ch_bw,debug=self.envDebug)
                self.channels.append(ch)
                src.out_channel.append(ch)
                self.channel_map[f_ctr] = [src.id, dest.id]
                f_ctr+=1
            elif 'PS' in str(edge[1]) and 'PG' not in str(edge[1]):
                src = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]
                dest = [x for x in self.nodes if x.id == edge[1]][0]
                src.out.append(dest)
                ch = CyberEnv.Channel(SimMan.env, 'CS', src = src, dest=dest,bw=self.ch_bw,debug=self.envDebug)
                self.channels.append(ch)
                src.out_channel.append(ch)
                self.channel_map[f_ctr] = [src.id, dest.id]
                f_ctr+=1

    def threat_model(self,targets):
        # at specific nodes we would modify the qlimit of the routers like reduce it and also change the channel bw capacity
        # router nodes to target
        router_ids_to_target = ['R7','R10','R11','R8','R4','R3']
        #r_ix = random.randint(0,len(router_ids_to_target)-1)

        r_ix = random.sample(range(0,len(router_ids_to_target)-1), targets)
        #print('***************************************')
        for r_i in r_ix: 
            rtr_to_target = [x for x in self.routers if x.id == router_ids_to_target[r_i]][0]

            """ for rtx_tg in router_ids_to_target:
                rtr_to_target = [x for x in self.routers if x.id == rtx_tg][0]
                rtr_to_target.qlimit = rtr_to_target.qlimit*0 """
            if rtr_to_target is not None:
                #print(rtr_to_target.id)
                rtr_to_target.qlimit = rtr_to_target.qlimit*0

        

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        #SimMan.init()
        #print('Resetting Env===============================================>')
        if self.deviceCount == 4:
            self.reinitialize_network()
        else:
            self.reinitialize_complex_network(self.G)
        #SimMan.runSimulation(5000)
        if self.with_threat:
            self.threat_model(random.randint(self.device_compromised,self.device_compromised))

        drop_rates = []
        for router in self.routers:
            drop_rates.append(router.packets_drop)

        channel_urs = []
        for channel in self.channels:
            channel_urs.append(channel.utilization_rate)
        if self.channelModel:
            drop_rates = channel_urs
        return np.array(drop_rates)
        
        #return random_val
    
    def step(self, action,result={}, pc_queue = Queue(), cp_queue=Queue()):
        assert self.action_space.contains(action)

        # extract the attribute from the action
        #routerIndex = action["device"]
        #next_hop = action["next_hop"]
        routerIndex = action[0]
        next_hop = action[1]

        # this block of code is considered if woth cyber and phy enivronment are played together
        if (pc_queue.empty() == False):
            # depending on the attrib lf and ss 
            val_phy_env = pc_queue.get()
            lf_s = val_phy_env['lf']
            ss = val_phy_env['ss'][-1] # last switch added

            # Add event at the specific PG depending on the Line fault and the current switching action taken
            for lf in lf_s:
                zone = self.comp_zones[lf]
                # based on the zone select the PG
                pg = self.senders[zone - 1]
                # trigger a single packet at this generator regarding the line fault info, usually it should be one time
                SimMan.process(pg.sendCommand())
                #SimMan.runSimulation(10)

            # get the pg to send the command for the switching
            pg_s = self.senders[self.comp_zones[ss] - 1]
            SimMan.process(pg_s.sendCommand())

        # create an event to send this action to the Simpy running environment
        #action_event = Message(StackMessageTypes.ASSIGN, {"device": routerIndex, "next_hop": next_hop})

        # got to implement how this action will be triggered in the simpy emulated env, based on the routerIndex obtain the device to update the 
        # get the router to update the routing policy
        selectedRouter = self.routers[routerIndex]

        # notify the router to trigger a process to change the routing path
        #selectedRouter.notifier.trigger(action_event)
        #next_hop_router = self.routers[next_hop]

        if next_hop < len(selectedRouter.out):
            SimMan.process(selectedRouter.change_receiver(next_hop))

        # Run the simulation until the assignment ends
        #SimMan.runSimulation(action_event)
        SimMan.runSimulation(50)
        result[0] = self.interpreter.getFeedback()
        # Return (observation, reward, done, info)

        return result[0]

    def render(self, mode='human', close=False):
        nx.draw_networkx(self.G, with_labels=True, pos=nx.spring_layout(self.G))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        #plt.savefig('cyber_network.png', dpi=150)
        pass
        
    
    



""" class CyberEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nodes, edges, simEnv):

        print('initializing the communication network')
        self.env = simEnv
        self.nodes = nodes 
        self.edges = edges
        self.G = self.constructNetwork()
        self.simTime = 10000
        # currently the action space will be discrete with network re-route. 
        # depending on the degree of the router node, the following action space would be defined. 
        # for instance the action would determine the next hop node priority 
        
        # We formulate two types of action in this work: re-routing and firewall policy update/ depending on the type of node
        # representing the action space in the form of dictionary
        # {'Firewall' : spaces.MultiDiscrete([no_of_firewalls, max_policy_count, 2])}
        # understanding the action space with an example, say there are 3 firewall and each firewall have say a maximum of 10 policies
        # a policy consist of {source n/w, destination n/w, and protocol type}
        # Then a sample action say [2, 7, 1] refers to 3rd Firewall , update the policy number 8 from deny to allow [Note : all action index start from 0]
        #  
        # {'Router' : spaces.MultiDiscrete([no_of_routers, no_of_dest_nw, max_no_of_priority])}
        # for instance, say a router X is connected to router A, B and C and for a specific destination, priority of router A > C > B 
        # suppose there are 5 router and say the network serves 10 destination network, and say each router have a max of 5 degree i.e. 5 max_no_of_priority
        # then a samoke action say [3, 2, 1] refers to the 4th router, for any packet coming to the destination network (3), 
        # should have the router number (2) as the next immediate hop [Note : all action index start from 0]


        n_actions = len()

        self.action_spaces = spaces.Discrete(n_actions)

        # observation space consists of number of dropped packets at nodes/ available bandwidth / available resource in the L3 device 
        # communication network topology
        self.observation_spaces = spaces.Box(low=0, high=1, shape=(len(self.G.Nodes) + len(self.G.Edges),), dtype=np.float32)

        print('Simpy Env initialized')

    def constructNetwork(self):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.id)
        for edge in self.edges:
            G.add_edge(edge[0].id,edge[1].id)
        return G

    # The action space 
    def step(self,action):

        # obtain the current state of the system

        # based on the event schedule an event in the current environment

        # obtain the current state and based on that obtain a reward associated with it

        # 
        NotImplementedError

    def reset(self):
        # restart the env everytime
        # Here the variability to be incorporated is to modify the packet arrival rate from one of the node


        self.env.run(until = self.simTime)
        

    def render(self):
        NotImplementedError """