from dp.launching.typing import BaseModel, Field
from dp.launching.typing import InputFilePath, OutputDirectory
from dp.launching.typing import Int, Float, List, Enum, String, Dict, Boolean, Optional
from dp.launching.typing.addon.sysmbol import Equal
import dp.launching.typing.addon.ui as ui
from dp.launching.typing import (
    BohriumUsername, 
    BohriumProjectId, 
    BohriumMachineType,
    BohriumTicket
)

from dp.launching.cli import (
    to_runner
)

style_group = ui.Group('Task Styles', 'Define type of teacher model, explore method and model distillation')
node_group = ui.Group('Default Setting', 'Setting default image and machine type')
conf_group = ui.Group('Structure Generation', 'Perturb initial structures')
explore_group = ui.Group('Exploration Setting','Set explore parameters')
model_group = ui.Group('Pretrained Model Style','Select pretrained model style')
task_group = ui.Group('Task Control','Job utility')
fp_group = ui.Group('DFT Calculation', 'Settings for DFT calculation')


class InjectConfig(BaseModel):
    '''
    Get bohrium username, etc.
    '''
    bohrium_username: BohriumUsername
    bohrium_ticket: BohriumTicket
    bohrium_project_id: BohriumProjectId

class UploadFiles(BaseModel):
    configurations: List[InputFilePath] = \
        Field(..., 
              title="initial configurations",
              description='Structural files of initial configurations (Must be in poscar format)')
    pretrained_model: InputFilePath = \
        Field(None, 
              title="pretrained model",
              max_file_count=1,
            description='Pretrained model file',
            )
    training_script: InputFilePath = \
        Field(None, ftypes=['json'], max_file_count=1,
              title="finetune training script",
                description='Training script for model finetune)',
        )
    fp_input_scf: InputFilePath = \
        Field(
            ..., 
            title="INPUT.scf",
            description="Input file for single-point DFT calculation"
        )
    fp_input_aimd: Optional[InputFilePath] = \
        Field (
            None, 
            title="INPUT.md",
            description="Input file for AIMD DFT calculation"
        )
    pp_files: List[InputFilePath] = \
        Field(..., 
              title="pseudo-potential files",
              description='Pseudopotential files')
    orb_files: Optional[List[InputFilePath]] = \
        Field(None,
              title="orbital files",
              description='Orbital basis files (Optional, only applies to ABACUS)')

class ModelStyleOptions(String, Enum):
    dp = "dp"

class ExploreStyleOptions(String,Enum):
    lmp_nvt='lmp-nvt'
    lmp_npt='lmp-npt'

class FpOptions(String,Enum):
    vasp="vasp"
    abacus="fpop_abacus"    

# explore options
class ExploreOptionsLAMMPS(String,Enum):
    npt="npt",
    nvt="nvt"

class ExploreConvergeType(String,Enum):
    energy_rmse="energy_rmse"
    force_rmse="force_rmse"
    
@node_group
class DefaultNode(BaseModel):
    default_machine: BohriumMachineType = Field(
        default=BohriumMachineType.C2_M8_CPU,
        title="default machine type"
    )
    default_image: String = Field(
        default="registry.dp.tech/dptech/ubuntu:22.04-py3.10",
        title="default image"
    )
    
@explore_group
class ExploreNode(BaseModel):
    explore_machine: String = Field(
        default="1 * NVIDIA V100_32g",
        title="machine type",
        index=10
    )
    explore_image: String = Field(
        default="registry.dp.tech/dptech/deepmd-kit:2024Q1-d23cf3e",
        title="image",
        index=11
    )
    group_size: Int = Field(
        default=1,
        title="group size",
        description="",
        index=12
    )
    pool_size: Int = Field(
        default=1,
        title="pool size",
        description="",
        index=13
    )
    
@model_group
class TrainNode(BaseModel):
    train_machine: String = Field(
        default="1 * NVIDIA V100_32g",
        title="default machine type"
    )
    train_image: String = Field(
        default="registry.dp.tech/dptech/deepmd-kit:2024Q1-d23cf3e"
    )

    
@model_group
class PretrainModel(BaseModel):
    pretrain_model_type: ModelStyleOptions = Field(
        default=ModelStyleOptions.dp,
        title="style",
        description="Teacher model style",
        index=0
    )
    custom_type_map : Boolean = Field(
        title="input type map",
        default=False,
        description="Using custom type map (The default map type is the same as that of DPA-2 Q1 model)",
        index=1
    )
    
@model_group
@ui.Visible(PretrainModel,"custom_type_map",Equal,True)
class TypeMap(BaseModel):
    type_map: String = Field(
        default=None,
        description="type map, e.g., 'H,Li,O'",
        index=2
    )
    
@conf_group    
class PertConf(BaseModel):
    atom_pert_distance: Float = Field(
        default=0.1,
        title="atom displacement",
        description="in Angstrom"
    )
    cell_pert_fraction: Float = Field(
        default=0.02,
        title="cell perturb fraction",
        description="relative displacement of lattice coordinate"
    )
    pert_num: Int = Field(
        default=100,
        title="pert number",
        description="Number of perturbed structure for each initial structure"
    )
    
# exploration
@explore_group
class ExploreParams(BaseModel):
    explore_style: ExploreStyleOptions = Field(
        default=ExploreStyleOptions.lmp_npt,
        description="Explore style",
        title="style",
        index=0
    )
    n_sample: Int = Field(
        default = 5,
        title="number of samples",
        description="Number of configurations selected for exploration",
        index=1
    )
    temps: List[Int] = Field(
        default=[300],
        description="Explore temperature",
        title="temperature",
        index=2
    )
    dt: Float = Field(
        default=0.002,
        title="dt",
        description="Timestep for MD exploration",
        index=4
    )
    nsteps: Int = Field(
        default=1000,
        description="Number of timesteps",
        title="number of step",
        index=5
    )
    trj_freq: Int = Field(
        default=10,
        title="trj_freq",
        description="Frequency of frame extraction",
        index=6
    )
    max_iter: Int = Field(
        default=1,
        title="max iteration",
        description="Max iteration. Ideally we should finish in one interation",
        index=7
    )
    converge_type: ExploreConvergeType = Field(
        default=ExploreConvergeType.force_rmse,
        title="converge criteria",
        description="The criteria for workflow convergence",
        index=8
    )
    converge_rmse: Float = Field(
        default=0.01,
        title="RMSE",
        index=9
    )
    
@explore_group
@ui.Visible(ExploreParams,"explore_style",Equal,"lmp-npt")
class ExploreParamsNptLAMMPS(BaseModel):
    press: List[Float] = Field(
        default=1.,
        description="Simulation pressure, in Bar",
        title="pressure",
        index=3
    )
    
@fp_group
class FpType(BaseModel):
    fp_type: FpOptions = Field(
        default=FpOptions.abacus,
        title="DFT routine",
        description="DFT software",
        index=0
    )
    
@fp_group
class FpMachine(BaseModel):
    fp_machine: String = Field(
        default="c32_m64_cpu",
        title="default machine type",
        index=3
    )
    
@fp_group
@ui.Visible(FpType,"fp_type",Equal,"fpop_abacus")
class FpImageABACUS(BaseModel):
    abacus_image: String = Field(
        default="registry.dp.tech/dptech/abacus:3.6.1",
        index=4
    )

@fp_group
@ui.Visible(FpType,"fp_type",Equal,"vasp")
class FpImageVASP(BaseModel):
    vasp_image: String = Field(
        default="registry.dp.tech/dptech/vasp:5.4.4",
        index=5
    )
    
    
@fp_group
@ui.Visible(FpType,"fp_type",Equal,"fpop_abacus")
class CommandABACUS(BaseModel):
    abacus_command: String = Field(
        default="OMP_NUM_THREADS=4 mpirun -np 8 abacus | tee log",
        title="command",
        description="Set the OpenMP thread according to machine type",
        index=2
    )

@fp_group
@ui.Visible(FpType,"fp_type",Equal,"vasp")
class CommandVASP(BaseModel):
    vasp_command: String = Field(
        default="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
        title="command OK",
        description="Set the OpenMP thread according to machine type",
        index=1
    )


@task_group
class Util(BaseModel):
    monitering: Boolean = Field(
        default=True,
        title="moniter job",
        description="The dflow job would be tracked in real time"
    ) 

class DemoMode(BaseModel):
    demo_mode: Boolean = Field(
        default= False,
        description="Use the predefined teacher model and configuration for demonstration"
    )

class Finetune(
    UploadFiles,
    InjectConfig,
    Util,
    CommandVASP,
    CommandABACUS,
    FpImageVASP,
    FpImageABACUS,
    FpMachine,
    FpType,
    TypeMap,
    ExploreParamsNptLAMMPS,
    ExploreParams,
    PertConf,
    PretrainModel,
    TrainNode,
    ExploreNode,
    DefaultNode,
    BaseModel
):
    output_directory: OutputDirectory = Field(default='./outputs')

def foo_runner(opts: Finetune):
        print("do nothing")

def to_parser():
        return to_runner(
            Finetune,foo_runner
    )


if __name__ == '__main__':
    import sys
    to_parser()(sys.argv[1:])