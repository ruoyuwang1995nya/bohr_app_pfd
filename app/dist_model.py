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
node_group = ui.Group('Node Selection', 'Setting image and machine type for computing nodes')
conf_group = ui.Group('Structure Generation', 'Perturb initial structures')
explore_group = ui.Group('Exploration Setting','Set explore parameters')
teacher_group = ui.Group('Teacher Model','Select teacher model style')
dist_group = ui.Group('Model Distillation','Select distillation model style')
task_group = ui.Group('Task Control','Job utility')

class InjectConfig(BaseModel):
    '''
    Get bohrium username, etc.
    '''
    # Bohrium config
    bohrium_username: BohriumUsername
    bohrium_ticket: BohriumTicket
    bohrium_project_id: BohriumProjectId

class UploadFiles(BaseModel):
    configurations: List[InputFilePath] = \
        Field(..., 
              title="initial configurations",
              description='Structural files of initial configurations (Must be in poscar format)')
    teacher_model_file: Optional[List[InputFilePath]] = \
        Field(None,
            title="teacher model", 
            description='Teacher model file (Do not upload if demo mode is selected)'
            )
    training_script: Optional[InputFilePath] = \
        Field(None, 
            title="distillation training script",
            ftypes=['json'], 
            description='(Optional) Training script for model distillation)',
        )

class TeacherModelStyleOptions(String, Enum):
    dp = "dp"

class ExploreStyleOptions(String,Enum):
    lmp_nvt='lmp-nvt'
    lmp_npt='lmp-npt'

class DistModelStyleOptions(String,Enum):
    dp="dp"

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
    
    
@teacher_group
class TeacherModel(BaseModel):
    teacher_model_type: TeacherModelStyleOptions = Field(
        default=TeacherModelStyleOptions.dp,
        title="style",
        description="Teacher model style"
    )
    custom_type_map : Boolean = Field(
        title="input type map",
        description="Using custom type map (The default map type is the same as that of DPA-2 Q1 model)",
        default=False
    )
    
@teacher_group
@ui.Visible(TeacherModel,"custom_type_map",Equal,True)
class TypeMap(BaseModel):
    type_map: String = Field(
        default=None,
        description="type map, e.g., 'H,Li,O'"
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
        default=10,
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
        default = 1,
        title="number of samples",
        description="Number of configurations selected for exploration"
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
        title="nstep",
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
        title="max iter",
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
    
@dist_group
class DistModel(BaseModel):
    dist_model_type: DistModelStyleOptions = Field(
        default=DistModelStyleOptions.dp,
        alias="style",
        description="Distillation model style",
        index=0
    )    
    
@dist_group
class TrainNode(BaseModel):
    train_machine: String = Field(
        default="1 * NVIDIA V100_32g",
        title="machine type",
        index=1
    )
    train_image: String = Field(
        title="image",
        default="registry.dp.tech/dptech/deepmd-kit:2024Q1-d23cf3e",
        index=2
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

class Dist(
    UploadFiles,
    InjectConfig,
    Util,
    TypeMap,
    DistModel,
    ExploreParamsNptLAMMPS,
    ExploreParams,
    PertConf,
    TeacherModel,
    TrainNode,
    ExploreNode,
    DefaultNode,
    BaseModel
):
    output_directory: OutputDirectory = Field(default='./outputs')

def foo_runner(opts: Dist):
        print("do nothing")

def to_parser():
        return to_runner(
            Dist,foo_runner
    )


if __name__ == '__main__':
    import sys
    to_parser()(sys.argv[1:])