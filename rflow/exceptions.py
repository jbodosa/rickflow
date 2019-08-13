# -*- coding: utf-8 -*-

"""
Exceptions for rickflow package
"""


class RickFlowException(Exception):
    pass


class LastSequenceReached(RickFlowException):
    def __init__(self, last_seqno):
        super(LastSequenceReached, self).__init__(
            "Last sequence {} reached. Quitting.".format(last_seqno)
        )


class NoCuda(RickFlowException):
    def __init__(self, exc):
        super(RickFlowException, self).__init__(
            "Cuda was required, but could not be used on the current node. "
            "Make sure that you have loaded the cuda module and that this "
            "script is running on a GPU node."
            "Details: {}".format(exc)
        )


class TrajectoryNotFound(RickFlowException):
    def __init__(self, traj_name):
        super(RickFlowException, self).__init__(
            "Trajectory {} could not be loaded".format(traj_name)
        )


class SoluteAtomsNotSet(RickFlowException):
    def __init__(self):
        super(RickFlowException, self).__init__(
            "solute_atoms not set. Please populate AlchemyFlow.solute_atoms "
            "before running an alchemical simulation."
        )
