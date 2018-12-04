# -*- coding: utf-8 -*-

import os
import inspect
import pkg_resources
import traceback


def abspath(relative_path): # type (object) -> object
    """Get file from a path that is relative to caller's module.
    Returns:    absolute path as string"""
    caller = inspect.stack()[1]
    mod = inspect.getmodule(caller[0])
    return os.path.normpath(pkg_resources.resource_filename(mod.__name__, relative_path))


class CWD(object):
    """
    change dir in a with block
    """
    def __init__(self, path): self.old_path = os.getcwd(); self.new_path = str(path)

    def __enter__(self): os.chdir(self.new_path); return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False # uncomment to pass exception through

        return True


def selection(trajectory, sel):
    """
    A short helper function to enable selection via atom ids or selection strings.

    Args:
        trajectory: A mdtraj trajectory.
        sel: Either a selection string or a list of atom ids.

    Returns:
        list of int: Selected atom ids.
    """
    if sel is None:
        return []
    elif isinstance(sel, list):
        return sel
    else:
        return trajectory.topology.select(sel)