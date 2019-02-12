import argparse
from utils import read_parameters
from string import Template

# Dict used to match job_type with name of python file
job_type2python_file = {'sampling': 'images_to_samples',
                        'training': 'train_model',
                        'inference': 'inference'}

# The following template is that of a shell script used to feed 'jobsub'
# Parameter values are taken from a yaml file
job_template = Template('''#!$shell_used
#$$ -N $job_name-$operation
#$$ -m $dash_m
#$$ -M $user_email
#
#$$ -j $join_stdout_err
#$$ -o $output_file
#
#$$ -S $shell_used
#$$ -pe $parallel_envs
#$$ -l res_cpus=$res_cpus
#$$ -l res_mem=$res_mem
#$$ -l h_rt=$wallclock_time
#$$ -l res_image=nrcan_all_default_ubuntu-14.04-amd64_latest
#$$ -l res_gputype=$res_gputype
#$$ -l res_gpus=$res_gpus
. ssmuse-sh -d /fs/ssm/nrcan/geobase/tensorflow
export PYTHONPATH=$python_envs/$default_env
export MKL_THREADING_LAYER=$mkl_threading_layer
source $python_envs/$default_env/bin/activate
conda activate $default_env
cd $work_base
python -u $user_home/$script_location/$operation.py $user_home/$operation_config_location/$config_filename
source deactivate''')


def main(job_type):
    """Generate runnable bash job file"""
    operation_dict = {'operation': job_type2python_file[job_type]}
    # Replace the undefined sections of parameters that are dependent on operation type
    params['output_file'] = params['output_file'].replace("$operation", operation_dict['operation'])
    # Concatenate operation_dict with params for final substitution
    params.update(operation_dict)
    shell_script = job_template.substitute(params)
    print(shell_script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='jobsub parameters')
    parser.add_argument('param_file', metavar='file',
                        help='Path to job parameters stored in yaml')
    parser.add_argument('job_type', metavar='Type of PyTorch job in [sampling, training, inference]',
                        help='Type of PyTorch job : sampling, training, inference',
                        choices=set(job_type2python_file.keys()))
    args = parser.parse_args()
    params = read_parameters(args.param_file)
    job_type = args.job_type

    main(params, job_type)
