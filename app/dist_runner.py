from pathlib import Path
from typing import List, Union
import os
import shutil
import json
import glob
from monty.serialization import loadfn
from .dist_model import Dist
from .constants import default_type_map,dist_train_script_template
from pfd.entrypoint.submit import FlowGen

def get_inputs(opts:Dist,
               model_path:Union[Path,str]
               ):
    inputs={}
    if opts.custom_type_map is True:
        inputs["type_map"]=str(opts.type_map).split(',')
    else:
        inputs["type_map"]=default_type_map
    inputs["teacher_models_paths"]=[model_path]
    return inputs
    
def get_train(opts:Dist,train_script):
    train={
        "type": "dp",
        "config": {
            "init_model_policy": "no",
            },
        "template_script":train_script
    }
    return train
    
def get_conf_generation(opts:Dist,
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

def get_exploration(opts:Dist):
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
        }
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

def get_global_config(opts: Dist):
    bohrium_config={
            "username": opts.bohrium_username,
            "ticket": opts.bohrium_ticket,
            "project_id": int(opts.bohrium_project_id)
        }
    return bohrium_config

def get_default_step_config(opts: Dist):
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

def get_run_train_config(opts: Dist):
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

def get_explore_config(opts: Dist):
    run_explore_config={
        "template_config": {
            "image": opts.explore_image
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
                        "scass_type": opts.explore_machine
                    }}}},
        "template_slice_config":{
                "group_size":opts.group_size,
                "pool_size":opts.pool_size
            }
    }
    return run_explore_config

def DistRunner(opts: Dist,
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

    # teacher model
    model_dir = workdir / "teacher_model"
    model_dir.mkdir()
    for ii in opts.teacher_model_file:
        shutil.copy(ii,model_dir)
    
    # train script
    if opts.training_script:
        print(opts.training_script)
        with open(opts.training_script,"r") as fp:
            train_script=json.load(fp)
    else:
        train_script=dist_train_script_template[opts.dist_model_type]
        
    # change to workdir
    output_dir=Path(opts.output_directory)
    #output_dir.mkdir()
    os.chdir(workdir)
    config={
        "bohrium_config":get_global_config(opts),
        "default_step_config":get_default_step_config(opts),
        "step_configs":{
            "run_train_config": get_run_train_config(opts),
            "run_explore_config": get_explore_config(opts)
        },
        "task":{"type":"dist"},
        "inputs":get_inputs(opts,glob.glob("teacher_model/*")[0]),
        "conf_generation": get_conf_generation(opts,
                            [ii for ii in glob.glob("confs/*")]),
        "train": get_train(opts,train_script),
        "exploration": get_exploration(opts)
    }
    
    with open("pfd_dist.json","w") as fp:
        json.dump(config,fp,indent=4)
        
    with open("pfd_dist.json","r") as fp:
        config=json.load(fp)
        
    # submit workflow
    FlowGen(config,download_path="./returns").submit(
        no_submission=no_submission,
        only_submit= not opts.monitering,
    )
    os.chdir(cwd)
    shutil.copytree(workdir, output_dir/'workdir', dirs_exist_ok = True)

