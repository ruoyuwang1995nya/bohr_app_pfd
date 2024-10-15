from pathlib import Path
from typing import List, Union, Optional, Dict
import os
import shutil
import glob
import json
from .ft_model import Finetune
from .util import get_element
from .constants import default_type_map,dist_train_script_template
from pfd.entrypoint.submit import FlowGen

def get_inputs(opts:Finetune
               ):
    inputs={}
    if opts.custom_type_map is True:
        inputs["type_map"]=opts.type_map
    else:
        inputs["type_map"]=default_type_map
    return inputs
    
def get_train(opts:Finetune,
              init_model: Union[Path,str],
              train_script: Union[Path,str]
              ):
    train={
        "type": "dp",
        "config": {
            "impl": "pytorch",
            "init_model_policy": "no",
            "init_model_with_finetune": True,
            },
        "init_models_paths":[init_model],
        "numb_models":1,
        "template_script":train_script
    }
    return train

def get_conf_generation(opts:Finetune,
                        confs: List[str]
                        ):
    conf_generation={
        "init_configurations":
            {
            "type": "file",
            "prefix": "./",
            "fmt": "vasp/poscar",
            "files": confs
            },
        "pert_generation":[
            {
                "conf_idx": "default",
                "atom_pert_distance":opts.atom_pert_distance,
                "cell_pert_fraction":opts.cell_pert_fraction,
                "pert_num": opts.pert_num
                }
            ]
        }
    return conf_generation

def get_exploration(opts:Finetune):
    if opts.explore_style in ["lmp-nvt","lmp-npt"]:
        exploration={
            "type": "lmp",
            "config": {
                "command": "lmp -var restart 0",
            },
            "stages":[],
        "max_iter":opts.max_iter,
        "converge_config":{
            "type":opts.converge_type,
            "RMSE":opts.converge_rmse
            },
        "test_set_config":{
            "test_size":0.1
        },
        "filter":[],
        }
        task={ 
                "conf_idx": [ii for ii in range(len(opts.configurations))],
                "n_sample":opts.n_sample,
                "exploration":{
                    "type": "lmp-md",
                    "dt":opts.dt,
                    "nsteps": opts.nsteps,
                    "temps": opts.temps,
                    "trj_freq": opts.trj_freq
                    },
                "max_sample": 10000
                }
        if opts.explore_style == "lmp-nvt":
            task["exploration"]["ensemble"]="nvt"
        elif opts.explore_style == "lmp-npt":
            task["exploration"]["ensemble"]="npt"
            task["exploration"]["press"]=opts.press
        exploration["stages"].append([task])
    else: 
        raise NotImplemented("Explore type not implemented!")
    return exploration 

def get_global_config(opts:Finetune):
    bohrium_config={
            "username": opts.bohrium_username,
            "ticket": opts.bohrium_ticket,
            "project_id": int(opts.bohrium_project_id)
        }
    return bohrium_config

def get_default_step_config(opts:Finetune):
    default_step_config={
        "template_config": {
            "image": opts.default_image
        },
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": opts.default_machine
                    }}}
        }
    }
    return default_step_config

def get_run_train_config(opts:Finetune):
    run_train_config={
        "template_config": {
            "image": opts.train_image
        },
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": opts.train_machine
                    }}}
        }
    }
    return run_train_config

def get_explore_config(opts:Finetune):
    run_explore_config={
        "template_config": {
            "image": opts.explore_image
        },
        "continue_on_success_ratio": 0.8,
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": opts.explore_machine
                    }}}},
        "template_slice_config":{
                "group_size":opts.group_size,
                "pool_size":opts.pool_size
            }
    }
    return run_explore_config

def get_fp_config(opts:Finetune):
    if opts.fp_type == "fpop_abacus":
        run_fp_config={
            "template_config": {
                "image": opts.abacus_image
            },
            "continue_on_success_ratio": 0.8,
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": opts.fp_machine
                            }
                        }
                    }
                }
    }
    elif opts.fp_type == "vasp":
        run_fp_config={
            "template_config": {
                "image": opts.vasp_image
            },
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": opts.fp_machine
                            }
                        }
                    }
                }
    }
    return run_fp_config

def get_fp(
    opts:Finetune,
    input_scf: Union[str,Path],
    pp_files: Dict,
    orb_files: Optional[Dict] = {}
    ):
    fp={
        "type":opts.fp_type,
        "extra_output_files:":[]
    }
    if opts.fp_type == "fpop_abacus":
        fp["run_config"]={
            "command":opts.abacus_command
        }
        fp["inputs_config"]={
            "input_file":input_scf,
            "pp_files": pp_files,
            "orb_files": orb_files
        }
    elif opts.fp_type == "vasp":
        fp["run_config"]={
            "command":opts.vasp_command
        }
        fp["inputs_config"]={
            "incar":input_scf,
            "pp_files": pp_files,
            "orb_files": orb_files
        }
    return fp

def FinetuneRunner(opts: Finetune,
                no_submission: bool =False
                ):
    cwd = Path.cwd()
    print('start running....')
    workdir = cwd / 'workdir'
    returns_dir = workdir / 'returns'
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    workdir.mkdir()
    returns_dir.mkdir()
    
    # copy configuration
    conf_dir = workdir / "confs"
    conf_dir.mkdir()
    for ii in opts.configurations:
        shutil.copy(ii, conf_dir)

    # pretrain model
    model_dir = workdir / "pretrain_model"
    model_dir.mkdir()
    shutil.copy(opts.pretrained_model,model_dir)
    shutil.copy(opts.training_script,workdir / "training_script.json")
    
    ## fp
    fp_dir = workdir / "fp"
    fp_dir.mkdir()
    shutil.copy(opts.fp_input_scf,fp_dir / "INPUT.scf")
    if opts.fp_input_aimd:
        shutil.copy(opts.fp_input_aimd,fp_dir / "INPUT.md")
    else:
        shutil.copy(opts.fp_input_scf,fp_dir / "INPUT.md")
        
    pp_dir = fp_dir / "pp"
    pp_dir.mkdir()
    for ii in opts.pp_files:
        shutil.copy(ii,pp_dir)
        
    if opts.orb_files:
        orb_dir = fp_dir / "orb"
        orb_dir.mkdir()
        for ii in opts.orb_files:
            shutil.copy(ii,orb_dir)
    
    output_dir=Path(opts.output_directory)
    output_dir.mkdir()
    # change to workdir
    os.chdir(workdir)
    
    pp_dict={}
    for ii in glob.glob("fp/pp/*"):
        pp_dict.update({get_element(ii):ii})
    
    orb_dict={}
    if opts.orb_files:
        for ii in glob.glob("fp/orb/*"):
            orb_dict.update({get_element(ii):ii})
        if not orb_dict.keys()==pp_dict.keys():
            raise KeyError("The Element of orb files must match that of pseudo-potential files exactly!")
    
    config={
        "bohrium_config":get_global_config(opts),
        "default_step_config":get_default_step_config(opts),
        "step_configs":{
            "run_train_config": get_run_train_config(opts),
            "run_explore_config": get_explore_config(opts),
            "run_fp_config":get_fp_config(opts)
        },
        "task":{
            "type":"finetune",
            "init_training":True,
            "skip_aimd":False
            },
        "inputs":get_inputs(opts),
        "conf_generation": get_conf_generation(opts,
                            [ii for ii in glob.glob("confs/*")]),
        "train": get_train(
            opts,
            glob.glob("pretrain_model/*")[0],
            "training_script.json"
            ),
        "exploration": get_exploration(opts),
        "fp": get_fp(opts,"./fp/INPUT.scf",pp_dict,orb_dict),
        "aimd":{
            "inputs_config": {
                "input_file": 
                    "./fp/INPUT.md"
                    }
            }
        }
    #print(config)
    with open("pfd_ft.json","w") as fp:
        json.dump(config,fp,indent=4)
        
    with open("pfd_ft.json","r") as fp:
        config=json.load(fp)

    # submit workflow
    FlowGen(config,download_path="./returns").submit(
        no_submission=no_submission,
        only_submit= not opts.monitering,
    )
    shutil.copytree(workdir, output_dir/'workdir', dirs_exist_ok = True)
