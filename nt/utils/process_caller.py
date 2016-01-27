from time import sleep
import subprocess
import numpy as np
from warnings import warn

DEBUG_MODE = False


def run_processes(cmds, sleep_time=0.1, ignore_return_code=False):
    """ Starts multiple processes, waits and returns the outputs when available

    :param cmd: A list with the commands to call
    :param sleep_time: Intervall to poll the running processes (in seconds)
    :param ignore_return_code: If true, ignores non zero return codes.
        Otherwise an exception is thrown.
    :return: Stdout, Stderr and return code for each process
    """

    # Ensure its a list if a single command is passed
    cmds = cmds if isinstance(cmds, list) else [cmds]

    if DEBUG_MODE:
        [print('Calling: {}'.format(cmd)) for cmd in cmds]
    pipes = [subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              shell=True,
                              universal_newlines=True) for cmd in cmds]
    return_codes = len(cmds) * [None]
    stdout = len(cmds) * [None]
    stderr = len(cmds) * [None]

    # Recover output as the processes finish
    for i, p in enumerate(pipes):
        stdout[i], stderr[i] = p.communicate()
        return_codes[i] = p.returncode

    for idx, code in enumerate(return_codes):
        txt = 'Command {} returned with return code {}.\n' \
                'Stdout: {}\n' \
                'Stderr: {}'.format(cmds[idx], code, stdout[idx], stderr[idx])
        if code != 0 and not ignore_return_code:
            raise EnvironmentError(txt)
        if code != 0 and ignore_return_code:
            warn('Returncode for command {} was {} but is ignored.\n'
                 'Stderr: {}'.format(
                cmds[idx], code, stderr[idx]))
        if DEBUG_MODE:
            print(txt)

    return stdout, stderr, return_codes
