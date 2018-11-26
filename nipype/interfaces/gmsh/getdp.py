from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    traits, TraitedSpec, isdefined, OutputMultiPath,
                                    File)
import os
import os.path as op
from nipype.utils.filemanip import split_filename


class GetDPInputSpec(CommandLineInputSpec):
    problem_file = File(exists=True, argstr="%s", position=1, mandatory=True)
    mesh_file = File(exists=True, mandatory=True, argstr="-msh %s",
                     desc="Read mesh (in Gmsh .msh format) from file")

    adapatation_constraint_file = File(
        exists=True, argstr="-adapt %s", desc="Read adaptation constraints from file")
    gmsh_read_file = File(exists=True, argstr="-gmshread %s",
                          desc="Read gmsh data (same as GmshRead in resolution)")
    results_file = File(exists=True, argstr="-res %s",
                        desc="Load processing results from file(s)")

    preprocessing_type = traits.String(argstr="-pre %s", desc="Pre-processing")
    run_processing = traits.Bool(argstr='-cal', desc="Run processing")
    postprocessing_type = traits.String(
        argstr="-post %s", desc="Post-processing")
    save_results_separately = traits.Bool(
        argstr='-split', desc="Save processing results in separate files")
    restart_processing = traits.Bool(
        argstr='-restart', desc="Resume processing from where it stopped")

    solve = traits.String(argstr="-solve %s",
                          desc="Solve (same as -pre 'Resolution' -cal)")

    output_name = traits.String("getdp", argstr="-name %s", usedefault=True,
                                desc="Generic file name")

    maximum_interpolation_order = traits.Int(argstr='-order %d',
                                             desc="Restrict maximum interpolation order")

    # Linear solver options
    binary_output_files = traits.Bool(
        argstr='-bin', desc="Create binary output files")
    mesh_based_output_files = traits.Bool(
        argstr='-v2', desc="Create mesh-based Gmsh output files when possible")

    out_table_filenames = traits.List(traits.Str, desc="List of table text files generated by the specified problem file. \
      If tables are written during the GetDP solving process, this is required for Nipype to find the proper output files for the interface.")

    out_pos_filenames = traits.List(traits.Str, desc="List of postprocessing (.pos) files generated by the specified \
      problem file. If tables are written during the GetDP solving process, this is required for Nipype to find the proper \
      output files for the interface.")


class GetDPOutputSpec(TraitedSpec):
    results_file = File(exists=True, desc='The generated results file')
    preprocessing_file = File(
        exists=True, desc='The generated preprocessing file')
    postprocessing_files = OutputMultiPath(
        File(exists=True), desc='Any generated postprocessing files')
    table_files = OutputMultiPath(
        File(exists=True), desc='Any generated postprocessing files')


class GetDP(CommandLine):
    """
    GetDP, a General environment for the treatment of Discrete Problems
    Copyright (C) 1997-2012 P. Dular, C. Geuzaine

    .. seealso::

    Gmsh

    Example
    -------

    >>> from nipype.interfaces.gmsh import GetDP
    >>> solve = GetDP()
    >>> solve.inputs.problem_file = 'eeg_forward.pro'
    >>> solve.inputs.mesh_file = 'mesh.msh'
    >>> solve.run()                                    # doctest: +SKIP
    """
    _cmd = 'getdp'
    input_spec = GetDPInputSpec
    output_spec = GetDPOutputSpec

    def _list_outputs(self):
        path, _, _ = split_filename(self.inputs.problem_file)
        outputs = self.output_spec().get()
        outputs['results_file'] = op.abspath(self._gen_outfilename())
        outputs['preprocessing_file'] = op.abspath(self._gen_outfilename())

        out_table_files = []
        for table_outfilename in self.inputs.out_table_filenames:
            _, name, _ = split_filename(table_outfilename)
            out_table_files.append(op.join(path, name + ".txt"))
        outputs['table_files'] = out_table_files

        out_pos_files = []
        for pos_outfilename in self.inputs.out_pos_filenames:
            _, name, _ = split_filename(pos_outfilename)
            out_pos_files.append(op.join(path, name + ".pos"))
        outputs['postprocessing_files'] = out_pos_files

        return outputs

    def _gen_outfilename(self, ext="res"):
        path, _, _ = split_filename(self.inputs.problem_file)
        _, name, _ = split_filename(self.inputs.output_name)
        return os.path.join(path, name + '.' + ext)