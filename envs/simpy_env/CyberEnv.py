# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:24:05 2022

@author: abhijeetsahu

This environment is created for episodic interaction of the Simpy based COmmunication simulator for training RL agent, similar to an Open AI Gym environment
"""
from multiprocessing import Event
from typing import Dict, List

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from envs.simpy_env.SimComponentsDynamic import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Router, Firewall,Packet
from envs.simpy_env.simtools import SimMan,Notifier
from abc import ABC, abstractmethod
import simpy
import random
import functools
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, List, Tuple
from queue import Queue

def constArrival():
    return 1.5    # time interval

def constSize():
    return 100.0  # bytes

adist = functools.partial(random.expovariate, 0.5)
sdist = functools.partial(random.expovariate, 0.01)  # mean size 100 bytes
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
        
        def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0):
            super(CyberEnv.SenderDevice, self).__init__(env, id,  adist, sdist)
            SimMan.process(self.senderProcess())
        
        def senderProcess(self):
            yield SimMan.timeout(self.initial_delay)
            while SimMan.now < self.finish:
                # wait for next transmission
                yield SimMan.timeout(self.adist())
                self.packets_sent += 1
                p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
                self.out.put(p)

        def sendCommand(self):
            yield SimMan.timeout(self.adist())
            self.packets_sent +=1
            p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            print(str(self.id) + ' : Sending control command')
            self.out.put(p)

    class ForwarderDevice(SwitchPort):
        def __init__(self, env,id, rate, qlimit=None, limit_bytes=True, debug=False):
            super(CyberEnv.ForwarderDevice, self).__init__(env,id, rate, qlimit, limit_bytes, debug)
            self.selected_Router_Index = 0
            SimMan.process(self.forwarderProcess())
            #self.notifier = Notifier('Updates', self)
        
        def forwarderProcess(self):
            while True:
                msg = (yield self.store.get())
                self.busy = 1
                self.byte_size -= msg.size
                yield SimMan.timeout(msg.size*8.0/self.rate)
                self.out[self.selected_Router_Index].put(msg)
                self.busy = 0
                if self.debug:
                    print(msg)

        # this function will change the receiver if it founds one receiver to be busy
        def change_receiver(self, new_receiver):
            #self.out = [new_receiver]
            self.selected_Router_Index = new_receiver
            print(str(self.id) + ' : Changing Route to ' + str(self.out[self.selected_Router_Index].id))
            
            yield SimMan.timeout(1)

    class ReceiverDevice(Interpreter):
        def __init__(self, simpyenv,id,gymenv, rec_arrivals=False, absolute_arrivals=False, rec_waits=True, debug=False, selector=None):
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
            self.bytes_rec = 0
            self.selector = selector
            self.last_arrival = 0.0
            self.reset()

        def put(self, pkt):
            if not self.selector or self.selector(pkt):
                now = self.env.now
                if self.rec_waits:
                    self.waits.append(self.env.now - pkt.time)
                if self.rec_arrivals:
                    if self.absolute_arrivals:
                        self.arrivals.append(now)
                    else:
                        self.arrivals.append(now - self.last_arrival)
                    self.last_arrival = now
                self.packets_rec += 1
                self.bytes_rec += pkt.size
                if self.debug:
                    print(pkt)
        
        def reset(self):
            self.receivedPackets = [0 for _ in range(len(self.gymenv.senders))]
            self._done = False

        def getReward(self):
            """
            Will depend on the percentage of successfully packet received
            """
            try:
                reward = self.packets_rec
                avg_wait = sum(self.waits)/len(self.waits)
                return float(reward / sum(self.waits))
                #return float(reward)
            except:
                return 0

        def getObservation(self):
            r_val = np.random.uniform(0,1000000.0,self.gymenv.deviceCount)

            # instead of random value lets  get some value
            r_val = []
            for router in self.gymenv.routers:
                r_val.append(router.packets_drop)
            return np.array(r_val)
        
        def getDone(self):
            total_loss = 0
            ctr = 0
            for rtrs in self.gymenv.routers:
                if rtrs.packets_rec > 0:
                    total_loss += float(rtrs.packets_drop)/rtrs.packets_rec
                    ctr+=1
            avg_lossrate = total_loss/ctr
            if avg_lossrate < 0.005:
                self._done = True
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
        def __init__(self,env, src,dest,bw=100, delay=10, snr = 10):
            self.env = env # simpy env
            self.src = src # source node
            self.dest = dest # destination node
            self.bw = bw # channel bandwidth
            self.delay = delay # channel delay (this should be propagation delay)
            self.channel_capacity = self.bw * math.log10(1 + snr) # shannon's channel capacity formula for noisy channel
            SimMan.process(self.run())

        def run(self):
            NotImplementedError
            # this function should start the channel and update the bandwidth and delay as the simulation progresses

            # cause latency

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

    def __init__(self, provided_graph= None):
        super(CyberEnv, self).__init__(deviceCount=4)

        self.comp_zones = {}
        if provided_graph is None:

            # Here we monitor the number of lost packets at all the router nodes (which is kind of the device count)
            self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(self.deviceCount,), dtype=np.float32)
            self.G = nx.Graph()
            self.reinitialize_network()

        else:
            self.deviceCount = 16
            self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(self.deviceCount,), dtype=np.float32)
            self.G = provided_graph
            self.reinitialize_complex_network(self.G)
            self.action_space = spaces.Dict({
            "device": spaces.Discrete(self.deviceCount),
            "next_hop": spaces.Discrete(self.deviceCount)
            })


    def reinitialize_network(self):
        SimMan.init()
        self.nodes = []
        self.senders: List[self.SenderDevice] = [
            CyberEnv.SenderDevice(SimMan.env, 'PG1', adist, sdist ),
            CyberEnv.SenderDevice(SimMan.env, 'PG2', constArrival, constSize)
        ]
        self.nodes.extend(self.senders)

        # initialize all the forwarding devices
        self.routers: List[self.ForwarderDevice] = [
            CyberEnv.ForwarderDevice(SimMan.env, 'R1',rate=400.0, qlimit=300,debug=False),
            CyberEnv.ForwarderDevice(SimMan.env,'R2',rate=400.0, qlimit=300,debug=False),
            CyberEnv.ForwarderDevice(SimMan.env,'R3',rate=300.0, qlimit=300,debug=False),
            CyberEnv.ForwarderDevice(SimMan.env,'R4',rate=300.0, qlimit=300,debug=False)
        ]
        self.nodes.extend(self.routers)

        self.interpreter = self.ReceiverDevice(SimMan.env, 'PS',self, debug=False)
        self.nodes.extend([self.interpreter])

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

        edges = [(self.senders[0],self.routers[0]), (self.senders[1],self.routers[1]), (self.routers[0],self.routers[1]), 
        (self.routers[0],self.routers[2]), (self.routers[1],self.routers[3]), (self.routers[2],self.routers[3]), (self.routers[3],self.interpreter)]

        for edge in edges:
            self.G.add_edge(edge[0].id,edge[1].id)


    # lets say we are given a graph we need to construct the simpy network for a larger system
    # G.add_node(n, nodetype='sender', 'router' or 'sink')
    def reinitialize_complex_network(self, G):
        SimMan.init()
        self.nodes = []
        self.senders = []
        self.routers = []
        self.interpreter = None
        for key,node in G.nodes(data=True):
            if node['nodetype'] == 'sender':
                g_node =  CyberEnv.SenderDevice(SimMan.env, key, adist, sdist )
                self.senders.append(g_node)
            elif node['nodetype'] == 'router':
                r_node = CyberEnv.ForwarderDevice(SimMan.env, 'R'+str(key),rate=400.0, qlimit=300,debug=False)
                self.routers.append(r_node)
            elif node['nodetype'] == 'sink':
                self.interpreter = self.ReceiverDevice(SimMan.env, key,self, debug=False)
        self.nodes.extend(self.senders)
        self.nodes.extend(self.routers)
        self.nodes.extend([self.interpreter])

        for edge in G.edges(data= True):
            if 'PG' in str(edge[1]):
                src = [x for x in self.nodes if x.id == edge[1]][0]
                dest = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]
                src.out = dest
            else:
                if 'PS' not in str(edge[1]): 
                    src = [x for x in self.nodes if x.id == 'R'+str(edge[1])][0]
                    dest = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]
                    if dest not in src.out:
                        src.out.append(dest)
                else:
                    src = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]
                    dest = [x for x in self.nodes if x.id == edge[1]][0]
                    src.out.append(dest)
                    

        """ for edge in G.edges(data=True):
            if 'P' in str(edge[0]):
                src = [x for x in self.nodes if x.id == edge[0]][0]
            else:
                src = [x for x in self.nodes if x.id == 'R'+str(edge[0])][0]

            if 'P' in str(edge[1]):
                dest = [x for x in self.nodes if x.id == edge[1]][0]
            else:
                dest = [x for x in self.nodes if x.id == 'R'+str(edge[1])][0]
            
            # if the source node is a sender, assuming every source has a single router and assuming the id of the router starts with 'R'
            if 'PG' in str(dest.id):
                dest.out = src
            elif 'PS' in str(dest.id):
                src.out.append(dest)
            else:
                # if the source node is a router
                src.out.append(dest)
                dest.out.append(src) """


    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        #SimMan.init()
        if self.deviceCount == 4:
            self.reinitialize_network()
        else:
            self.reinitialize_complex_network(self.G)
        SimMan.runSimulation(10)
        random_val = np.random.uniform(0,1000000.0,self.deviceCount)
        random_val = []
        for router in self.routers:
            random_val.append(router.packets_drop)
        return np.array(random_val)
        
        #return random_val
    
    def step(self, action, result, pc_queue = Queue(), cp_queue=Queue()):
        assert self.action_space.contains(action)

        # extract the attribute from the action
        routerIndex = action["device"]
        next_hop = action["next_hop"]
    
        #print(' received data from phy env : '+str(pc_queue.get()))

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
        #SimMan.runSimulation(10)

        # create an event to send this action to the Simpy running environment
        #action_event = Message(StackMessageTypes.ASSIGN, {"device": routerIndex, "next_hop": next_hop})

        # got to implement how this action will be triggered in the simpy emulated env, based on the routerIndex obtain the device to update the 
        # get the router to update the routing policy
        selectedRouter = self.routers[routerIndex]

        # notify the router to trigger a process to change the routing path
        #selectedRouter.notifier.trigger(action_event)
        """ if self.deviceCount == 4:
            next_hop_router = self.routers[next_hop+1]
        else: """
        #next_hop_router = self.routers[next_hop]

        SimMan.process(selectedRouter.change_receiver(next_hop))
        #SimMan.process(selectedRouter.change_receiver(next_hop_router))

        # Run the simulation until the assignment ends
        #SimMan.runSimulation(action_event)
        SimMan.runSimulation(1000)
        result[0] = self.interpreter.getFeedback()

        # Return (observation, reward, done, info)
        return self.interpreter.getFeedback(),result

    def render(self, mode='human', close=False):
        nx.draw_networkx(self.G, with_labels=True, pos=nx.spring_layout(self.G))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        #plt.savefig('cyber_network.png', dpi=150)
        pass
        
    
    