#!/usr/bin/env python
"""Run a full ``YamboWannier90WorkChain``.

Usage: ./example_complete_hBN.py

To compare bands between QE, W90, W90 with QP, run in terminal
```
aiida-yambo-wannier90 plot bands PW_PK GWW90_PK
```
Where `PW_PK` is the PK of a `PwBandsWorkChain/PwBaseWorkChain` for PW bands calculation,
`GWW90_PK` is the PK of a `YamboWannier90WorkChain`.
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder 
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization, set_num_bands
from aiida_wannier90_workflows.utils.workflows.builder.submit import submit_and_add_group 
from aiida_wannier90_workflows.common.types import WannierProjectionType
from ase.io import read

def submit(group: orm.Group = None, run: bool = False, projections="atomic_projectors_qe"):
    """Submit a ``YamboWannier90WorkChain`` from scratch.

    projections can be "analytic" or "atomic_projectors_qe".
    
    Run all the steps.
    """
    # pylint: disable=import-outside-toplevel
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
    from aiida_wannier90_workflows.workflows.base.wannier90 import (
        Wannier90BaseWorkChain,
    )

    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    codes = {
        #"pw": "pw-7.1@hydralogin",
        #"pw2wannier90": "pw2wannier90-7.1@hydralogin",
        "pw": "pw-6.8-p2w_ham@hydralogin",
        "pw2wannier90": "pw2wannier90-6.8-p2w_ham@hydralogin",
        "projwfc": "projwfc-7.1@hydralogin",
        "wannier90": "w90@hydralogin",
        "yambo": "yambo-5.1@hydralogin",
        "p2y": "p2y-5.1@hydralogin",
        "ypp": "ypp-5.1@hydralogin",
        "gw2wannier90": "gw2wannier90@hydralogin",
    }

    ase_Si = read("./input_files/example_01/Cu.cif")
    structure = orm.StructureData(ase=ase_Si)
    
    wannier_projection_type=WannierProjectionType.ANALYTIC if projections == "analytic" else WannierProjectionType.ATOMIC_PROJECTORS_QE 

    builder = YamboWannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        pseudo_family="PseudoDojo/0.4/PBE/SR/standard/upf",
        protocol="fast",
        wannier_projection_type=wannier_projection_type,
    )

    # Increase ecutwfc, to have hig
    params = builder.yambo.ywfl.scf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 100
    builder.yambo.ywfl.scf.pw.parameters = orm.Dict(dict=params)
    params = builder.yambo.ywfl.nscf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 100
    builder.yambo.ywfl.nscf.pw.parameters = orm.Dict(dict=params)

    parallelization = dict(
        max_wallclock_seconds=24 * 3600,
        # num_mpiprocs_per_machine=48,
        #npool=4,
        num_machines=1,
    )
    set_parallelization(
        builder["yambo"]["ywfl"]["scf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["yambo"]["ywfl"]["nscf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["yambo_qp"]["scf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["yambo_qp"]["nscf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )

    set_parallelization(
        builder["wannier90"],
        parallelization=parallelization,
        process_class=Wannier90BandsWorkChain,
    )
    set_parallelization(
        builder["wannier90_qp"],
        parallelization=parallelization,
        process_class=Wannier90BaseWorkChain,
    )

    '''builder['yambo']['parameters_space']= orm.List(list=[{'conv_thr': 1,
                                 'conv_thr_units': '%',
                                 'convergence_algorithm': 'new_algorithm_1D',
                                 'delta': [2, 2, 2],
                                 'max': [32, 32, 32],
                                 'max_iterations': 4,
                                 'start': [8, 8, 8],
                                 'steps': 4,
                                 'stop': [16, 16, 16],
                                 'var': ['kpoint_mesh']},])'''
    
    builder['yambo']['workflow_settings']= orm.Dict(dict= {'bands_nscf_update': 'all-at-once',
                                 'skip_pre': True,
                                 'type': '1D_convergence',
                                 'what': ['gap_GG']},)


    
    #### START computational resources settings.
    builder['yambo']['ywfl']['yres']['yambo']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                                'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    builder['yambo']['ywfl']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
        
    builder['yambo']['ywfl']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 2, 'num_mpiprocs_per_machine':8}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=2'}}


    builder['yambo_qp']['additional_parsing'] = orm.List(list=['gap_GG'])

    builder['yambo_qp']['yres']['yambo']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                                'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    builder['yambo_qp']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
        
    builder['yambo_qp']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    
    builder['wannier90_qp']['wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
    
    
    
    builder['wannier90']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines':1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}


    builder['wannier90']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
    
    builder['wannier90']['pw2wannier90']['pw2wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 8, 'num_mpiprocs_per_machine':2}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=8'}}
    
    builder['wannier90']['wannier90']['wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc':2, 'num_mpiprocs_per_machine': 8}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=2'}}

    preprend_ypp_w = builder['ypp']['ypp']['metadata']['options']['prepend_text']
    builder['ypp']['ypp']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': preprend_ypp_w}}
    
    preprend_ypp_qp = builder['ypp_QP']['ypp']['metadata']['options']['prepend_text']
    builder['ypp_QP']['ypp']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': preprend_ypp_qp}}
    
    builder['gw2wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines':1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                            'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    #### END computational resources settings.
     
     
    # SKIP Convergence:
    # 1. pop "yambo"
    # 2. pop the "parent_folder" of "yambo_qp"
    # 3. add the "GW_mesh"  KpointsData input.
    
    builder.pop('yambo') # to skip the convergence
    builder['yambo_qp'].pop('parent_folder',None) # to skip the convergence
    
    
    # SKIP the yambo QP step:
    # this will skip yambo_qp, but be sure to provide QP_DB and parent_folder to ypp inputs.
    # In general, you can do this if you have already the yambo results.
    #builder.pop('yambo_qp')
    #builder.ypp.ypp.QP_DB = orm.load_node(13233)
    #builder.ypp.parent_folder = orm.load_node(13069).outputs.remote_folder
    
    # SET custom K-MESH:
    kpoints = orm.KpointsData() # to skip the convergence
    kpoints.set_kpoints_mesh([8,8,8]) # to skip the convergence
    builder.GW_mesh = kpoints # to skip the convergence
    
    #If we want W90 to use the GW mesh a priori, set the following to True. Usually, if converged for GW, it should be ok also for the Wannierization.
    builder.kpoints_force_gw = orm.Bool(True)
    
    # START projections settings:
    
    set_num_bands(
        builder=builder.wannier90, 
        num_bands=20,                   # KS states used in the Wannierization
        exclude_bands=range(1,5), 
        process_class=Wannier90BandsWorkChain)
    
    params = builder.wannier90.wannier90.wannier90.parameters.get_dict()

    ## START explicit atomic projections:
    if projections=="analytic":
    
        del builder.wannier90.projwfc
        builder.wannier90.wannier90.auto_energy_windows = False
        builder.wannier90.wannier90.shift_energy_windows = True
        params['num_wann'] = 9
        builder.wannier90.wannier90.wannier90.projections = orm.List(list=['Cu:s', 'Cu:p', 'Cu:d',])
        builder.wannier90_qp.wannier90.projections = builder.wannier90.wannier90.wannier90.projections
        params.pop('auto_projections', None) # Uncomment this if you want analytic atomic projections

        #
        # The following line can be also deleted.
        builder['wannier90']['pw2wannier90']['pw2wannier90']['parameters'] = orm.Dict(dict={'inputpp': {'atom_proj': False}})
    
    ## END explicit atomic projections:
    
    # optional settings.
    #params.pop('dis_proj_min', None)
    #params.pop('dis_proj_max', None)
    #params['num_wann'] = 16
    #params['dis_froz_max'] = 2
    
    params = orm.Dict(dict=params)
    builder.wannier90.wannier90.wannier90.parameters = params
    builder.wannier90_qp.wannier90.parameters = params
    
    # END projections settings.
    
    # START QP settings:
    #builder['yambo_qp']['parent_folder'] = orm.load_node(13004).outputs.remote_folder
    if "yambo_qp" in builder.keys():
        builder['yambo_qp']['QP_subset_dict'] = orm.Dict(dict={
                                            'qp_per_subset':50,
                                            'parallel_runs':4,
                                    })

        builder['yambo_qp']['yres']['yambo']['parameters'] = orm.Dict(dict={'arguments': ['dipoles', 'ppa', 'HF_and_locXC', 'gw0', 'rim_cut'],
            'variables': {'Chimod': 'hartree',
            'DysSolver': 'n',
            'GTermKind': 'BG',
            'PAR_def_mode': 'workload',
            'X_and_IO_nCPU_LinAlg_INV': [1, ''],
            'RandQpts': [5000000, ''],
            'RandGvec': [100, 'RL'],
            'NGsBlkXp': [2, 'Ry'],
            'FFTGvecs': [20, 'Ry'],
            'BndsRnXp': [[1, 80], ''],
            'GbndRnge': [[1, 80], ''],
            'QPkrange': [[[1, 1, 32, 32]], '']}})
    
    # END QP settings.

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.GROUP(
    help="The group to add the submitted workchain.",
)
@RUN()
def cli(group, run):
    """Run a ``YamboWannier90WorkChain``."""
    submit(group, run, projections="analytic")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter