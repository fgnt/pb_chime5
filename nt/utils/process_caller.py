from time import sleep
import subprocess
import numpy as np
from warnings import warn
import os

DEBUG_MODE = False
DEFAULT_ENV = os.environ.copy()


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        # Improve error msg with stdout and stderr
        return '{}\n\nStdout:\n{}\n\nStderr:\n{}'.format(
            super().__str__(), self.stdout, self.stderr
        )


def run_process(
        cmd,
        *,
        shell=None,
        check=True,
        environment=None,
        cwd=None,
        input=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
):
    """
    This is a wrapper around subprocess.run with changed defaults.
     - stdout and stderr is captured
     - output is checked
     - shell is True when cmd is a string
        - The shell need to split the command
        - Wildcard support
        - environment variable support
     - universal_newlines:
        - https://stackoverflow.com/a/38182530
        - Enable text_mode i.e. use strings and not bytes

    ToDo:
        Add a tee argument and display the stdout and stderr immediately and
        return the output. Need test for input, example password ask.
        Inspiration with aysncio: https://stackoverflow.com/a/25755038/5766934

    cmd: str or list of [string, bytes, os.PathLike]

    universal_newlines:
        stdout will be a string and not a byte array
    shell:
        If None, shell is True is cmd is a str else False
        True: pass command through the shell with "shell" parsing.
            i.e. wildcards, enviroments variables, etc.
        False: Directly called, recommended, when cmd is a list, because when
            for example strings contains whitespaces they are not interpreted.

    # Effect of universal_newlines
    >>> run_process('echo Hello')
    CompletedProcess(args='echo Hello', returncode=0, stdout='Hello\\n', stderr='')
    >>> run_process('echo Hello', universal_newlines=False)
    CompletedProcess(args='echo Hello', returncode=0, stdout=b'Hello\\n', stderr=b'')

    # Example, where shell=False is usefull
    >>> run_process(['echo', 'Hello World'])
    CompletedProcess(args=['echo', 'Hello World'], returncode=0, stdout='Hello World\\n', stderr='')
    >>> run_process(['echo', 'Hello World'], shell=True)
    CompletedProcess(args=['echo', 'Hello World'], returncode=0, stdout='\\n', stderr='')
    >>> run_process(['echo', 'Hello World'], shell=False)
    CompletedProcess(args=['echo', 'Hello World'], returncode=0, stdout='Hello World\\n', stderr='')


    """

    if shell is None:
        if isinstance(cmd, str):
            shell = True
        else:
            shell = False

    try:
        return subprocess.run(
            cmd,
            input=input,
            universal_newlines=universal_newlines,
            shell=shell,
            stdout=stdout,
            stderr=stderr,
            check=check,
            env=environment,
            cwd=cwd
        )
    except subprocess.CalledProcessError as e:
        # Improve error msg with stdout and stderr
        raise CalledProcessError(
            returncode=e.returncode,
            cmd=e.cmd,
            output=e.output,
            stderr=e.stderr
        ) from e


def run_processes(cmds, sleep_time=None, ignore_return_code=False,
                  environment=DEFAULT_ENV, warn_on_ignore=True,
                  inputs=None, *, cwd=None, shell=True):
    """ Starts multiple processes, waits and returns the outputs when available

    :param cmd: A list with the commands to call
    :param sleep_time: Intervall to poll the running processes (in seconds)
        This option is deprecated, the implementation does not need a
        sleep_time anymore.
    :param ignore_return_code: If true, ignores non zero return codes.
        Otherwise an exception is thrown.
    :param environment: environment (e.g. path variable) for commands
    :param warn_on_ignore: warn if return code is ignored but non zero
    :param inputs: A list with the text inputs to be piped to the called commands
    :return: Stdout, Stderr and return code for each process
    """
    # ToDo: remove sleep_time
    if sleep_time is not None:
        import warnings
        warnings.warn('Call nt.utils.process_caller.run_processes(..., '
                      'sleep_time={}, ...) is deprecated. sleep_time will be '
                      'removed in the future'.format(sleep_time))

    # Ensure its a list if a single command is passed
    cmds = cmds if isinstance(cmds, list) else [cmds]
    if inputs is None:
        inputs = len(cmds) * [None]
    else:
        inputs = inputs if isinstance(inputs, list) else [inputs]

    if DEBUG_MODE:
        [print('Calling: {}'.format(cmd)) for cmd in cmds]
    try:
        pipes = [subprocess.Popen(cmd,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=shell,
                                  universal_newlines=True,
                                  cwd=cwd,
                                  env=environment) for cmd in cmds]
    except Exception:
        print('cmds:', cmds)
        print('cwd:', cwd)
        raise
    return_codes = len(cmds) * [None]
    stdout = len(cmds) * [None]
    stderr = len(cmds) * [None]

    # Recover output as the processes finish
    for i, p in enumerate(pipes):
        stdout[i], stderr[i] = p.communicate(inputs[i])
        return_codes[i] = p.returncode

    raise_error_txt = ''
    for idx, code in enumerate(return_codes):
        txt = 'Command {} returned with return code {}.\n' \
                'Stdout: {}\n' \
                'Stderr: {}'.format(cmds[idx], code, stdout[idx], stderr[idx])
        if code != 0 and not ignore_return_code:
            raise_error_txt += txt + '\n'
        if code != 0 and ignore_return_code and warn_on_ignore:
            warn('Returncode for command {} was {} but is ignored.\n'
                 'Stderr: {}'.format(
                cmds[idx], code, stderr[idx]))
        if DEBUG_MODE:
            print(txt)
    if raise_error_txt != '':
        raise EnvironmentError(raise_error_txt)

    return stdout, stderr, return_codes
