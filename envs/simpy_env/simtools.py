from simpy import Environment
from simpy.events import Event, Process
from simpy.rt import RealtimeEnvironment
import logging as logger
from typing import (Any, Callable, DefaultDict, Dict, Generator, List, Set,
                    Tuple, Union)

from collections import defaultdict, deque
import itertools
from numbers import Number

class SimulationManager:
    """
    The :class:`SimulationManager` offers methods and properties for managing
    and accessing a SimPy simulation.
    Note:
        Do not create instances on your own. Reference the existing instance by
        :attr:`SimMan` instead.
    """
    
    def __init__(self):
        self._env = None
    
    @property
    def env(self):
        """
        The SimPy :class:`~simpy.core.Environment` object belonging to the
        current simulation
        """
        return self._env

    @env.setter
    def env(self, environment):
        logger.debug("SimulationManager: Setting environment")
        self.env = environment
    
    def nextTimeSlot(self, timeSlotLength: float) -> Event:
        """
        Returns a SimPy timeout event that is scheduled for the beginning of the
        next time slot. A time slot starts whenever `now` % `timeSlotLength` is
        ``0``.
        Args:
            timeSlotLength: The time slot length in seconds
        """
        return self.timeout(timeSlotLength - (self.now % timeSlotLength))

    @property
    def now(self):
        """int: The current simulation time step"""
        return self.env.now

    def process(self, generator: Generator[Event, None, None]) -> Process:
        """
        Registers a SimPy process generator (a generator yielding SimPy events)
        at the SimPy environment and returns it.
        Args:
            process: The generator to be registered as a process
        """
        return self.env.process(generator)

    def event(self):
        """
        Creates and returns a new :class:`~simpy.events.Event` object belonging to the
        current environment.
        """
        return Event(self.env)

    def runSimulation(self, until: Union[int, float, Event]):
        """
        Runs the simulation (or continues running it) until the amount of
        simulated time specified by `until` has passed (with `until` being a
        :class:`float`) or `until` is triggered (with `until` being an
        :class:`Event`).
        """
        logger.info("SimulationManager: Running simulation...")
        if not isinstance(until, Event):
            assert isinstance(until, Number)
            until = self.now + until
        self.env.run(until)
    
    def init(self):
        """
        Creates a new :class:`~simpy.core.Environment`.
        """
        logger.debug("SimulationManager: Initializing environment")
        self._env = Environment()
    
    def timeout(self, duration: float, value: Any = None) -> Event:
        """
        Shorthand for env.timeout(duration, value)
        """
        return self.env.timeout(duration, value)
    
    def timeoutUntil(self, triggerTime: float, value: Any = None) -> Event:
        """
        Returns a SimPy :class:`~simpy.events.EventEvent` that succeeds at the
        simulated time specified by `triggerTime`.
        Args:
            triggerTime: When to trigger the :class:`~simpy.events.Event`
            value: The value to call :meth:`~simpy.events.Event.succeed` with
        """
        now = self.now
        if triggerTime > now:
            return self.timeout(triggerTime-now, value)
        else:
            return self.timeout(0, value)
    
    def triggerAfterTimeout(self, event: Event, timeout: float, value: Any = None):
        """
        Calls :meth:`~simpy.events.Event.succeed` on the `event` after the
        simulated time specified in `timeout` has passed. If the event has
        already been triggered by then, no action is taken.
        """
        def callback(caller):
            if not event.triggered:
                event.succeed(value)
        timeoutEvent = self.timeout(timeout)
        timeoutEvent.callbacks.append(callback)

    # got to implement the change route login
    def updateRoute(self):
        NotImplementedError


def ownerPrefix(ownerObject: Any) -> str:
    """
    Calls :meth:`__repr__` on the `ownerObject` (if it is not ``None``) and
    returns the result concatenated with '.'.
    If the object is ``None``, an empty string will be returned.
    """
    if ownerObject is None:
        return ''
    return repr(ownerObject) + '.'

def strAndRepr(obj: Any) -> str:
    """
    Returns "str (repr)" where str and repr are the result of `str(obj)` and
    `repr(obj)`.
    """
    return "{} ({})".format(str(obj), repr(obj))


class Notifier:
    """
    A class that helps implementing the observer pattern. A :class:`Notifier`
    can be triggered providing a value. Both callback functions and SimPy
    generators can be subscribed. Every time the :class:`Notifier` is triggered,
    it will run its callback methods and trigger the execution of the subscribed
    SimPy generators. Aditionally, SimPy generators can wait for a
    :class:`Notifier` to be triggered by yielding its :attr:`event`.
    """

    def __init__(self, name: str = "", owner: Any = None):
        """
        Args:
            name: A string to identify the :class:`Notifier` instance (e.g.
                among all other :class:`Notifier` instances of the owner object)
            owner: The object that provides the :class:`Notifier` instance
        """
        self._name = name
        self._owner = owner
        self._event = None

        # Callbacks
        # A priority -> Set[Callable] defaultdict for callbacks:
        self._priorityToCallbacks: DefaultDict[int, Set[Callable[[Any], None]]] = defaultdict(set)
        self._callbackToPriority: Dict[Callable[[Any], None], int] = {}
        self._callbackToAdditionalArgs: Dict[Callable[[Any], None], Any] = {}
        self._sortedCallbacks = [] # List of callbacks sorted by their priority

        # SimPy generators
        self._processExecutors = {}

    def subscribeCallback(self, callback: Callable[[Any], None], priority: int = 0, additionalArgs: List[Any] = None):
        """
        Adds the passed callable to the set of callback functions. Thus, when
        the :class:`Notifier` gets triggered, the callable will be invoked
        passing the value that the :class:`Notifier` was triggered with.
        Note:
            A callable can only be added once, regardless of its priority.
        Args:
            callback: The callable to be subscribed
            priority: If set, the callable is guaranteed to be invoked only
                after every callback with a higher priority value has been executed.
                Callbacks added without a priority value are assumed to have
                priority `0`.
            additionalArgs: A list of arguments that are passed as further
                arguments when the callback function is invoked
        """
        # Every callback is only allowed to be added once
        assert callback not in self._callbackToPriority
        assert callback not in self._callbackToAdditionalArgs

        self._callbackToPriority[callback] = priority
        if additionalArgs is not None:
            self._callbackToAdditionalArgs[callback] = additionalArgs

        # Add the callback to the set belonging to its priority
        self._priorityToCallbacks[priority].add(callback)
        self._updateSortedCallbacks()
    
    def unsubscribeCallback(self, callback: Callable[[Any], None]):
        """
        Removes the passed callable from the set of callback functions.
        Afterwards, it is not triggered anymore by this :class:`Notifier`.
        Args:
            callback: The callable to be removed
        """
        # The given callback has to be subscribed
        assert callback in self._callbackToPriority

        priority = self._callbackToPriority.pop(callback)
        self._priorityToCallbacks[priority].remove(callback)
        self._updateSortedCallbacks()

        if callback in self._callbackToAdditionalArgs:
            self._callbackToAdditionalArgs.pop(callback)
    
    def _updateSortedCallbacks(self):
        """
        Rebuilds the :attr:`_sortedCallbacks` list
        """
        sortedPriorities = sorted(self._priorityToCallbacks.keys(), reverse=True)
        self._sortedCallbacks = list(
            itertools.chain(
                *[self._priorityToCallbacks[p] for p in sortedPriorities]
            )
        )

    def subscribeProcess(self, process: Generator[Event, Any, None], blocking=True, queued=False):
        """
        Makes the SimPy environment process the passed generator function when
        :meth:`trigger` is called. The value passed to :meth:`trigger` will also
        be passed to the generator function.
        Args:
            blocking: If set to ``False``, only one instance of the generator
                will be processed at a time. Thus, if :meth:`trigger` is called
                while the SimPy process started by an earlier :meth:`trigger` call
                has not terminated, no action is taken.
            queued: Only relevant if blocking is ``True``. If `queued` is set to
                false ``False``, a :meth:`trigger` call while an instance of the
                generator is still active will not result in an additional generator
                execution. If queued is set to ``True`` instead, the values of
                :meth:`trigger` calls that happen while the subscribed generator is
                being processed will be queued and as long as the queue is not
                empty, a new generator instance with a queued value will be
                processed every time a previous instance has terminated.
        """

        if process in self._processExecutors:
            logger.warn("Generator function %s is already subscribed! Ignoring the call.", process, sender=self)
        else:
            # creating an executor function that will be called whenever trigger is called
            def executor(value: Any):
                if not blocking:
                    # start a new process
                    SimMan.process(process(value))
                else:
                    if executor.running:
                        if not queued:
                            logger.debug("Notification with object '%s' did not lead "
                                            "to the processing of %s, since a previous SimPy "
                                            "process is still active, 'blocking' is 'True', and "
                                            "'queued' is 'False'.", value, process, sender=self)
                        else:
                            if len(executor.queue) == 10000:
                                    logger.critical("Queue of subscribed SimPy generator method %s reached a "
                                                    "size of 10000. Is this intended?", process, sender=self)
                            executor.queue.append(value)
                            logger.debug("Object '%s' was appended to queue for SimPy generator method %s, "
                                        "since a previous SimPy process is still active. "
                                        "Queue length: %d", value, process, len(executor.queue), sender=self)
                    else:
                        executor.running = True
                        processedEvent = SimMan.process(process(value))
                        if queued:
                            def executeNext(prevProcessReturnValue: Any):
                                # callback for running the next process from the queue
                                if len(executor.queue) > 0:
                                    nextObject = executor.queue.popleft()
                                    logger.debug("Processing generator %s "
                                                "with queued object %s.", process, nextObject, sender=self)
                                    event = SimMan.process(process(nextObject))
                                    event.callbacks.append(executeNext)
                                else:
                                    executor.running = False
                                    logger.debug("All queued jobs for generator %s were executed.", process, sender=self)
                            processedEvent.callbacks.append(executeNext)
                        else:
                            # We have to set executor.running back to False,
                            # once the generator has been processed.
                            def setRunningFlagToFalse(prevProcessReturnValue: Any):
                                executor.running = False
                            processedEvent.callbacks.append(setRunningFlagToFalse)

            executor.running = False
            if blocking:
                executor.queue = deque()
            self._processExecutors[process] = executor
    
    def trigger(self, value: Any = None):
        """
        Triggers the :class:`Notifier`. This runs the callbacks, makes the
        :attr:`event` succeed, and triggers the processing of subscribed SimPy
        generators.
        """
        logger.debug("Triggered with value %s", value, sender=self)
        for callback in self._sortedCallbacks:
            if callback in self._callbackToAdditionalArgs:
                args = self._callbackToAdditionalArgs[callback]
                callback(value, *args)
            else:
                callback(value)
            
        for executor in self._processExecutors.values():
            executor(value)
        if self._event is not None:
            self._event.succeed(value)
            self._event = None
    
    @property
    def event(self):
        """
        :class:`~simpy.events.Event`: A SimPy event that succeeds when the
        :class:`Notifier` is triggered
        """
        if self._event is None:
            self._event = SimMan.event()
        return self._event
    
    @property
    def name(self):
        """
        str: The :class:`Notifier`'s name as it has been passed to the constructor
        """
        return self._name
    
    def __repr__(self):
        return "{}Notifier('{}')".format(ownerPrefix(self._owner), self.name)

SimMan = SimulationManager()