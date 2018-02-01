import re
import tempfile
from collections import namedtuple
from math import log
from os import path, remove, rename, stat
from nt.kaldi.helper import excute_kaldi_commands
from nt.visualization import PDF as _PDF


State = namedtuple('State', ['name', 'arcs', 'id'])
FinalState = namedtuple('FinalState', ['state_id', 'weight'])
Arc = namedtuple('Arc', ['src', 'dst', 'ilabel', 'olabel', 'weight'])


def _add_input_output(input_files=None, output_file=None, pipe=True):
    """ Add input and output files to command in the form
    'command input_files > output_file', 'command > output_file',
    'command input_files |', 'command input_files', 'command |' or 'command'
    depending on the given files and the value of pipe

    :param input_files: list of file(s) to read from (None: stdin)
    :param output_file: file to write to (None: pipe or stdout,
                        depending on value of pipe)
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :return: string with input and output modifiers
    """

    input_files =\
        input_files if isinstance(input_files, (list, tuple)) else [
            input_files]

    cmd = ''
    for input_file in input_files:
        if input_file:
            cmd += ' {}'.format(input_file)

    if output_file:
        cmd += ' > {}'.format(output_file)
    elif pipe:
        cmd += ' | '

    return cmd


def fstcompile_cmd(in_txt=None, out_fst=None, isym_table=None, osym_table=None,
                   pipe=True, **kwargs):
    """ Generates a command to compile a fst from text format into binary format

    :param in_txt: input fst in text format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param isym_table: input symbol table file
    :param osym_table: output symbol table file
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :param kwargs: see fstpostprocess_cmd
    :return: string with fstcompile command and parameters
    """

    cmd = 'fstcompile'
    if isym_table:
        cmd += ' --isymbols={}'.format(isym_table)
    if osym_table:
        cmd += ' --osymbols={}'.format(osym_table)

    cmd += _add_input_output(in_txt)
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def fstaddselfloops_cmd(in_fst=None, out_fst=None, disambig_in=0,
                        disambig_out=0, pipe=True, **kwargs):
    """ Generates a command to add self loops to the fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param disambig_in: list of input symbols
    :param disambig_out: list of corresponding output symbols
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :param kwargs: see fstpostprocess_cmd
    :return: string with fstaddselfloops command and parameters
    """

    cmd = 'fstaddselfloops'
    disambig_in = disambig_in if isinstance(disambig_in, (list, tuple))\
        else [disambig_in]
    disambig_out = disambig_out if isinstance(disambig_out, (list, tuple))\
        else [disambig_out]
    fmt = ' "echo ' + '{} ' * len(disambig_in) + '|"'
    cmd += fmt.format(*disambig_in)
    cmd += fmt.format(*disambig_out)
    cmd += _add_input_output(in_fst)
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def fstrmsymbols_cmd(in_fst=None, out_fst=None, in_symbols=0,
                     out_symbols=0, pipe=True, **kwargs):
    """ Generates a command to remove symbols from the fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param in_symbols: list of input symbols to remove
    :param out_symbols: list of output symbols to remove
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :param kwargs: see fstpostprocess_cmd
    :return: string with fstrmsymbols command and parameters
    """

    cmd = ''
    if not in_symbols and not out_symbols:
        cmd += 'cat'

    if in_symbols:
        cmd = 'fstrmsymbols'
        in_symbols = in_symbols if isinstance(in_symbols, (list, tuple))\
            else [in_symbols]
        fmt = ' "echo ' + '{} ' * len(in_symbols) + '|"'
        cmd += fmt.format(*in_symbols)
        if out_symbols:
            cmd += _add_input_output(in_fst, pipe=True)
            in_fst = None

    if out_symbols:
        cmd += 'fstrmsymbols --remove-from-output=true'
        out_symbols = out_symbols if isinstance(out_symbols, (list, tuple))\
            else [out_symbols]
        fmt = ' "echo ' + '{} ' * len(out_symbols) + '|"'
        cmd += fmt.format(*out_symbols)

    cmd += _add_input_output(in_fst)
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def fstshortestpath_cmd(in_fst=None, out_fst=None, nshortest=1,
                        pipe=True, **kwargs):
    """ Generates a command to find shortest path(s) in fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param nshortest: number of shortest paths
    :param pipe: only if output file is None: True: pipe, False: stdout
    :param kwargs: see fstpostprocess_cmd
    :return: string with fstshortestpath command and parameters
    """

    cmd = 'fstshortestpath'
    cmd += ' --nshortest={}'.format(nshortest)
    cmd += _add_input_output(in_fst)
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def fstrandgen_cmd(in_fst=None, out_fst=None, select='log_prob',
                   npath=1, pipe=True, **kwargs):
    """ Generates a command to randomly generate fsts

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param select: arc selector for random generation:
                   'log_prob' (treat weights as negative log prob),
                   'uniform'  (draw uniformly)
    :param npath: number of paths to generate
    :param pipe: only if output file is None: True: pipe, False: stdout
    :param kwargs: see fstpostprocess_cmd
    :return: string with fstrandgen command and parameters
    """

    cmd = 'fstrandgen'
    cmd += ' --npath={}'.format(npath)
    cmd += ' --select={}'.format(select)
    cmd += _add_input_output(in_fst)
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def fstprint_cmd(in_fst=None, out_txt=None, isym_table=None,
                 osym_table=None, pipe=True):
    """ Generates a command to print a fst from binary format into text format

    :param in_fst: input fst in binary format (default: stdin)
    :param out_txt: output fst in text format (default: pipe)
    :param isym_table: input symbol table file
    :param osym_table: output symbol table file
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :return: string with fstcompile command and parameters
    """

    cmd = 'fstprint'
    if isym_table:
        cmd += ' --isymbols={}'.format(isym_table)
    if osym_table:
        cmd += ' --osymbols={}'.format(osym_table)

    cmd += _add_input_output(in_fst, out_txt, pipe=pipe)

    return cmd


def fstdeterminize_cmd(in_fst=None, out_fst=None, use_log=True, pipe=True):
    """ Generates a command to determinize a fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param use_log: determinize in log semiring
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :return: string with fstdeterminize command and parameters
    """

    cmd = 'fstdeterminizestar'
    if use_log:
        cmd += ' --use-log=true'

    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fstminimize_cmd(in_fst=None, out_fst=None, pipe=True):
    """ Generates a command to minimize a fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param pipe: only used if output file is None: True: write to pipe,
                 False: write to stdout
    :return: string with fstminimize command and parameters
    """

    cmd = 'fstminimizeencoded'
    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fstrmepsilon_cmd(in_fst=None, out_fst=None, pipe=True):
    """ Generates a command to remove epsilon transitions from a fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param pipe: only if output file is None: True: pipe, False: stdout
    :return: string with fstrmepsilon command and parameters
    """

    cmd = 'fstrmepsilon'
    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fsttopsort_cmd(in_fst=None, out_fst=None, pipe=True):
    """ Generates a command to topologically sort fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param pipe: only if output file is None: True: pipe, False: stdout
    :return: string with fsttopsort command and parameters
    """

    cmd = 'fsttopsort'
    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fstproject_cmd(in_fst=None, out_fst=None, project_output=False, pipe=True):
    """ Generates a command to project onto input or output labels of fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param project_output: False: project labels from input,
                           True: project labels from output
    :param pipe: only if output file is None: True: pipe, False: stdout
    :return: string with fstproject command and parameters
    """

    cmd = 'fstproject'
    if project_output:
        cmd += ' --project_output=true'
    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fstarcsort_cmd(in_fst=None, out_fst=None, sort_type='ilabel', pipe=True):
    """ Generates a command to sort the arcs in an fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param sort_type: sort type, 'ilabel' or 'olabel'
    :param pipe: only if output file is None: True: pipe, False: stdout
    :return: string with fstarcsort command and parameters
    """

    cmd = 'fstarcsort'
    if sort_type not in ['ilabel', 'olabel']:
        raise Exception('Unknown sort_type {}!'.format(sort_type))

    cmd += ' --sort_type={}'.format(sort_type)
    cmd += _add_input_output(in_fst, out_fst, pipe)

    return cmd


def fstpostprocess_cmd(in_fst=None, out_fst=None, project=False,
                       project_output=False, determinize=False, use_log=True,
                       minimize=False, rmepsilon=False, topsort=False,
                       arcsort=True, sort_type='ilabel', pipe=True):
    """ Generate postprocessing command to apply sevral operations to fst

    :param in_fst: input fst in binary format (default: stdin)
    :param out_fst: output fst in binary format (default: pipe)
    :param project: project on input or ouput labels
    :param project_output: False: project labels on input,
                           True: project labels on output
    :param determinize: determinize fst
    :param use_log: determinize in logartihmic semiring
    :param minimize: minimize fst
    :param rmepsilon: remove epsilon paths
    :param topsort: do topological sort
    :param arcsort: sort arcs
    :param sort_type: sort type: 'ilabel' or 'olabel'
    :param pipe: only if output file is None: True: pipe, False: stdout
    :return: string with postprocessing commands
    """

    num_cmds = sum([project, determinize, minimize,
                    rmepsilon, topsort, arcsort])
    cmd = ''
    if num_cmds == 0:
        cmd += 'cat'
    if project:
        num_cmds -= 1
        cmd += fstproject_cmd(in_fst, project_output=project_output,
                              pipe=(num_cmds > 0))
        in_fst = None
    if determinize:
        num_cmds -= 1
        cmd += fstdeterminize_cmd(in_fst, use_log=use_log, pipe=(num_cmds > 0))
        in_fst = None
    if minimize:
        num_cmds -= 1
        cmd += fstminimize_cmd(in_fst, pipe=(num_cmds > 0))
        in_fst = None
    if rmepsilon:
        num_cmds -= 1
        cmd += fstrmepsilon_cmd(in_fst, pipe=(num_cmds > 0))
        in_fst = None
    if topsort:
        num_cmds -= 1
        cmd += fsttopsort_cmd(in_fst, pipe=(num_cmds > 0))
        in_fst = None
    if arcsort:
        num_cmds -= 1
        cmd += fstarcsort_cmd(in_fst, sort_type=sort_type, pipe=(num_cmds > 0))
        in_fst = None

    cmd += _add_input_output(in_fst, out_fst, pipe=pipe)

    return cmd


def fstcompose_cmd(left_fst=None, right_fst=None, out_fst=None,
                   phi=None, pipe=True, **kwargs):
    """ Generates a command to compse two fsts

    :param left_fst: right fst in binary format (default: stdin)
    :param right_fst: left fst in binary format
    :param out_fst: output fst in binary format (default: pipe)
    :param pipe: only if output file is None: True: pipe, False: stdout
    :param phi: phi symbol to be used for phi composition
    :param kwargs: see fstpostprocess_cmd
    :return: string with fsttablecompose/fstphicompose command and parameters
    """

    if not phi:
        cmd = 'fsttablecompose'
    else:
        cmd = 'fstphicompose {}'.format(phi)

    if not left_fst:
        if right_fst:
            left_fst = '-'
        else:
            raise Exception('Either right or left fst has to be specified!')

    if not right_fst:
        right_fst = '-'

    cmd += _add_input_output([left_fst, right_fst])
    cmd += fstpostprocess_cmd(None, out_fst, pipe=pipe, **kwargs)

    return cmd


def build_from_txt(transducer_as_txt, output_file, isym_table=None,
                   osym_table=None, determinize=True, minimize=True,
                   addselfloops=False, disambig_in=0, disambig_out=0,
                   rmepsilon=False, sort_type="ilabel", input_as_txt=None):
    """ build transducer from text file or text input

    :param transducer_as_txt: input fst in text format
    :param output_file: output fst in binary format
    :param isym_table: input symbol table file
    :param osym_table: output symbol table file
    :param determinize: determinize fst
    :param minimize: minimize fst
    :param addselfloops: add self loops to fst
    :param disambig_in: list of input symbols
    :param disambig_out: list of corresponding output symbols
    :param rmepsilon: rmepsilons
    :param sort_type: sort type - ilabel or olabel
    :param input_as_txt: optional input in text format
                         (only used if transducer_as_txt is None)
    """

    if addselfloops:
        cmd = fstcompile_cmd(transducer_as_txt, isym_table=isym_table,
                             osym_table=osym_table, determinize=determinize,
                             minimize=minimize, arcsort=False)
        cmd += fstaddselfloops_cmd(out_fst=output_file, disambig_in=disambig_in,
                                   disambig_out=disambig_out,
                                   rmepsilon=rmepsilon, sort_type=sort_type)
    else:
        cmd = fstcompile_cmd(transducer_as_txt, output_file, isym_table,
                             osym_table, determinize=determinize,
                             minimize=minimize, rmepsilon=rmepsilon,
                             sort_type=sort_type)

    excute_kaldi_commands(cmd, inputs=input_as_txt)


def compose(fst1, fst2, output_file, phi=None, **kwargs):
    """ compse two fsts

    :param fst1: right fst in binary format
    :param fst2: left fst in binary format
    :param output_file: output fst in binary format
    :param phi: phi symbol to be used for phi composition
    :param kwargs: see fstpostprocess_cmd
    """

    cmd = fstcompose_cmd(fst1, fst2, output_file, phi, **kwargs)
    excute_kaldi_commands(cmd)


def shortestpath(fst, output_file, nshortest=1, **kwargs):
    """ Find shortet path through fst

    :param fst: input fst in binary format
    :param output_file: output fst in binary format
    :param nshortest: number of shortest paths
    :param kwargs: see fstpostprocess_cmd
    """

    cmd = fstshortestpath_cmd(fst, output_file, nshortest, **kwargs)
    excute_kaldi_commands(cmd)


def randgen(fst, output_file, select='log_prob', npath=1, **kwargs):
    """ Randomly generate path through given fst

    :param fst: input fst in binary format
    :param output_file: output fst in binary format
    :param select: arc selector for random generation:
                   'log_prob' (treat weights as negative log prob),
                   'uniform'  (draw uniformly)
    :param npath: number of paths to generate
    :param kwargs: see fstpostprocess_cmd
    """

    cmd = fstrandgen_cmd(fst, output_file, select, npath, **kwargs)
    excute_kaldi_commands(cmd)


def arcsort(fst, sort_type='ilabel'):
    """ Sort a given fst

    :param fst: fst to sort in binary format
    :param sort_type: sort type: ilabel or olabel
    """

    fst_tmp = path.join(path.dirname(fst), 'fst.tmp')
    cmd = fstarcsort_cmd(fst, fst_tmp, sort_type)
    excute_kaldi_commands(cmd)
    rename(fst_tmp, fst)


def check_fst_valid(fst, fast=True):
    if fast:
        return path.exists(fst) and (stat(fst).st_size > 0)
    cmd = 'fstinfo {}'.format(fst)
    stdout, stderr, return_codes = excute_kaldi_commands(
        cmd, ignore_return_code=True
    )
    if return_codes[0] != 0:
        return False
    else:
        return True


def remove_oovs(grammar_path, oov_list, new_path=None):
    with open(grammar_path) as fid:
        lines = fid.readlines()
        lines = [line for line in lines
                 if not [word for word in line.split()[2:4]
                         if word in oov_list]]

    if not new_path:
        new_path = grammar_path
    with open(new_path, 'w') as fid:
        fid.writelines(lines)


def remove_disambig_symbols(fst1, fst2, special_symbol_ids, sort_type=None,
                            minimize=True):
    with tempfile.NamedTemporaryFile() as disambig_list:
        with open(disambig_list.name, 'w') as fid:
            for s_id in special_symbol_ids:
                if s_id is not None:
                    fid.write('{}\n'.format(s_id))
        if minimize:
            min_cmd = ' | fstminimizeencoded'
        else:
            min_cmd = ''
        if sort_type is not None:
            arcsort_cmd = ' | fstarcsort --sort_type={} '.format(sort_type)
        else:
            arcsort_cmd = ''
        cmd = 'cat {} | fstrmsymbols {} | fstrmepslocal' \
              '{}{} > {}'.format(fst1, disambig_list.name,
                                 min_cmd, arcsort_cmd, fst2)
        excute_kaldi_commands(cmd)


def replace_labels(grammar_path, label_mapping, new_path=None, on_input=True,
                   on_output=True):
    def get_repl_func(word):
        def _repl(matchobj):
            prefix = matchobj.group(1)
            suffix = matchobj.group(2)
            return prefix + label_mapping[word] + suffix
        return _repl

    with open(grammar_path) as fid:
        out = fid.read()
        for word in label_mapping:
            if on_input:
                out = re.sub("^(\d+\s+\d+\s+)" + word + "(\s+)",
                             get_repl_func(word), out, flags=re.M)
            if on_output:
                out = re.sub("^(\d+\s+\d+\s+[^\s]+\s+)" + word + "(\s+)",
                             get_repl_func(word), out, flags=re.M)

    if not new_path:
        new_path = grammar_path
    with open(new_path, 'w') as fid:
        fid.write(out)


def map_labels_to_ints(grammar_path, label_mapping, new_path=None,
                       on_input=True, on_output=True):

    lines = list()
    with open(grammar_path) as fid:
        for line in fid.readlines():
            line = line.split()
            if len(line) >= 4:
                label_in = line[2]
                label_out = line[3]
                if on_input and label_in in label_mapping:
                    line[2] = str(label_mapping[line[2]])
                if on_output and label_out in label_mapping:
                    line[3] = str(label_mapping[line[3]])
            line = "\t".join(line) + "\n"
            lines.append(line)

    if not new_path:
        new_path = grammar_path
    with open(new_path, 'w') as fid:
        fid.writelines(lines)


def draw(isym_table, osym_table, fst_file, output_file):
    """ draw fst as pdf

    :param isym_table: input symbol table file
    :param osym_table: output symbol table file
    :param fst_file: input fst in binary format
    :param output_file: ouput pdf file
    """
    cmd = 'fstdraw --portrait=true --height=17 --width=22'
    if isym_table is not None:
        cmd += ' --isymbols={}'.format(isym_table)
    if osym_table is not None:
        cmd += ' --osymbols={}'.format(osym_table)
    cmd += ' {}'.format(fst_file)
    excute_kaldi_commands(
        cmd + ' > {}'.format(output_file.replace('pdf', 'dot')))

    cmd += ' | dot -Tpdf > {}'.format(output_file)

    excute_kaldi_commands(cmd)


def print_pdf(fst_file, isym_table, osym_table):
    """ print fst as pdf in ipython notebook

    :param fst_file: input fst in binary format
    :param isym_table: input symbol table file
    :param osym_table: output symbol table file
    :return: PDF object to display in notebook
    """
    draw(isym_table, osym_table, fst_file, fst_file + '.pdf')
    return _PDF(fst_file + '.pdf')


def draw_search_graph(search_graph, isym_table, osym_table):
    with tempfile.NamedTemporaryFile() as tmp_graph:
        cmd = 'lattice-to-fst ark:{} ark,t:- | tail -n +2 > {}'.format(
            search_graph, tmp_graph.name
        )
        excute_kaldi_commands(cmd)
        with tempfile.NamedTemporaryFile() as search_fst:
            build_from_txt(tmp_graph.name, search_fst.name, determinize=False,
                           minimize=False)
            draw(isym_table, osym_table, search_fst.name, search_graph + '.pdf')


def to_lattice(src_dir, fst, lattice, cost_type="acoustic"):
    fst_tmp_as_txt_path = path.join(src_dir, "fst_as_text_tmp.txt")
    cmd = "fstprint " + fst + "> " + fst_tmp_as_txt_path
    excute_kaldi_commands(cmd)
    fst_tmp_as_txt_fid = open(fst_tmp_as_txt_path, 'r')
    lattice_path = path.join(src_dir, lattice)
    lattice_fid = open(lattice_path, 'w')
    lattice_fid.write("UTT_ID \n")

    if cost_type not in ["acoustic", "lm"]:
        print("unknown sort_type: using default (ilabel)")

    for line in fst_tmp_as_txt_fid:
        columns = line.split()
        if len(columns) == 5:
            if cost_type == "acoustic":
                lattice_fid.write(
                    "{0}\t{1}\t{2}\t{3}\t0,{4}\n".format(columns[0], columns[1],
                                                         columns[2], columns[3],
                                                         columns[4]))
            else:
                lattice_fid.write(
                    "{0}\t{1}\t{2}\t{3}\t{4},0\n".format(columns[0], columns[1],
                                                         columns[2], columns[3],
                                                         columns[4]))
        elif len(columns) == 4:
            lattice_fid.write(
                "{0}\t{1}\t{2}\t{3}\n".format(columns[0], columns[1],
                                              columns[2], columns[3]))
        else:
            lattice_fid.write("{0}\n".format(columns[0]))

        del columns

    lattice_fid.write("\n")
    fst_tmp_as_txt_fid.close()
    lattice_fid.close()
    remove(fst_tmp_as_txt_path)


class SimpleFST:
    """
    Build, store and write a simple FST
    """

    def __init__(self):
        """
        Construct empty FST
        """

        self.states = list()
        self.final_states = list()
        self.start_state = None

    def add_state(self, name=''):
        """
        Add state to FST (states get integer state ids assigned, stating from 0)

        :param name: name assigned to the state
        :return: integer state id
        """

        self.states.append(State(name, list(), len(self.states)))
        return len(self.states) - 1

    def add_arc(self, src, dst, ilabel, olabel, weight=None, mode='always'):
        """
        Add arc to state src going to state dst with input label ilabel,
        output label olabel and weight weight.

        :param src: source state id
        :param dst: destination state id
        :param ilabel: input label
        :param olabel: output label
        :param weight: weight for transition
        :param mode: how to add arcs:
                     'always: always add arcs, even if duplicated
                     'if_not_exists': only add if arc dows not exist
        """

        assert src < len(self.states)
        assert dst < len(self.states)
        if mode == 'if_not_exists':
            if self.find_arc(src, ilabel, olabel, dst):
                return

        self.states[src].arcs.append(Arc(src, dst, ilabel, olabel, weight))

    def set_final(self, state_id, weight=None):
        """
        Set a state being a final state. Multiple states can be final.
        Call this function for each state separately

        :param state_id: id of state to be final
        :param weight: weight of state to be final
        """

        assert state_id < len(self.states)
        self.final_states.append(FinalState(state_id, weight))

    def set_start(self, state_id):
        """
        Set a state being the start state. Only one state can be a start state.
        The start state is changed with each call.

        :param state_id: id of state to be start state
        """

        assert state_id < len(self.states)
        self.start_state = state_id

    def find_arc(self, src, ilabel=None, olabel=None, dst=None):
        """
        Find last arc originating from state src with input label ilabel
        and output label olabel going to state dst

        :param src: source state id
        :param ilabel: input label
        :param olabel: output label
        :param dst: destination state id
        :return: Arc object instance
        """

        assert src < len(self.states)
        for arc in reversed(self.states[src].arcs):
            if ((ilabel is None) or (arc.ilabel == ilabel)) and\
                    ((olabel is None) or (arc.olabel == olabel)) and\
                    ((dst is None) or (arc.dst == dst)):
                return arc

    def get_arcs(self, src):
        """
        Get all arcs originating from state src

        :param src: source state id
        :return: list of Arc object instances
        """

        assert src < len(self.states)
        return self.states[src].arcs

    def add_self_loops(self, eps, ilabel, olabel, mode='before'):
        """
        Add self loops to state if it has outgoing (mode='before')
        or incoming (mode='after') transitions not having
        eps as an output label.

        :param eps: skip adding self loop if state has only olabel transitions
                    beeing eps
        :param ilabel: input label to add
        :param olabel: ouput label to add
        :param mode: 'before': add before transition (current state),
                     'after': add after transition (next state)
        """

        num_states = self.num_states
        for idx_state in range(num_states):
            for arc in self.get_arcs(idx_state):
                if arc.olabel != eps:
                    if mode == 'before':
                        self.add_arc(idx_state, idx_state, ilabel,
                                     olabel, mode='if_not_exists')
                        break
                    if mode == 'after':
                        self.add_arc(arc.dst, arc.dst, ilabel,
                                     olabel, mode='if_not_exists')

    def get_txt(self):
        """
        Get text version of fst

        :return: text version of fst
        """
        txt_arc_list = list()
        if self.start_state is not None:
            txt_arc_list.append(
                self._get_txt_arcs(self.states[self.start_state].arcs))
        for state in self.states:
            if state.id != self.start_state:
                txt_arc_list.append(self._get_txt_arcs(state.arcs))
        for final_state in self.final_states:
            txt_arc_list.append(self._get_txt_arcs(final_state))

        return '\n'.join(txt_arc_list)

    def write_txt(self, filename):
        """
        Write FST in text format

        :param filename: text file to write to
        """

        with open(filename, 'w') as fid:
            fid.write(self.get_txt())

    def write_fst(self, filename, determinize=False, minimize=True,
                  addselfloops=False, disambig_in=0, disambig_out=0,
                  rmepsilon=False, sort_type='ilabel', isyms=None,
                  osyms=None):
        """
        Write FST in openfst format.

        :param filename: filename to write to
        :param determinize: determinize written fst
        :param minimize: minimize written fst
        :param addselfloops: add self loops to each state with emmiting symbols
                             on at least one output arc
        :param disambig_in: filename of input disabiguity smols (list of ints)
        :param disambig_out: filename of output disabiguity smols (list of ints)
        :param rmepsilon: remove epsilons
        :param sort_type: soft FST, e.g. 'ilabel' or 'olabel'
        :param isyms: filename of input symbols mapping (symbol id)
        :param osyms: filename of output symbols mapping (symbol id)
        """

        with tempfile.NamedTemporaryFile() as fst_txt:
            self.write_txt(fst_txt.name)
            build_from_txt(
                fst_txt.name, filename, isym_table=isyms, osym_table=osyms,
                determinize=determinize, minimize=minimize,
                addselfloops=addselfloops, disambig_in=disambig_in,
                disambig_out=disambig_out, rmepsilon=rmepsilon,
                sort_type=sort_type)

    def get_state_name(self, state_id):
        """
        return name of state with id state id

        :param state_id: state id
        :return: state name
        """

        return self.states[state_id].name

    @property
    def num_states(self):
        """
        Return number of states

        :return: Number of states
        """

        return len(self.states)

    @property
    def num_arcs(self):
        """
        Return number of arcs

        :return: Number of arcs
        """

        return sum(len(state.arcs) for state in self.states)

    @staticmethod
    def _get_txt_arcs(arcs):
        """
        Get arcs in txt format

        :param arcs: List of Arc object instances or FinalState object instaces
        :return: arcs in txt format
        """

        arcs = arcs if isinstance(arcs, list) else [arcs]
        text_arc_list = list()
        for arc in arcs:
            if isinstance(arc, Arc):
                if arc.weight:
                    text_arc_list.append('{}\t{}\t{}\t{}\t{}'.format(*arc))
                else:
                    text_arc_list.append('{}\t{}\t{}\t{}'.format(*arc))
            if isinstance(arc, FinalState):
                if arc.weight:
                    text_arc_list.append('{}\t{}'.format(*arc))
                else:
                    text_arc_list.append('{}'.format(*arc))
            if not isinstance(arc, (Arc, FinalState)):
                raise Exception("Wrong arc not of type 'Arc' or 'FinalState'")

        return '\n'.join(text_arc_list)
