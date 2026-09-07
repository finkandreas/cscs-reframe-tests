import sys
import pathlib
import reframe as rfm
import reframe.utility.sanity as sn


from pytorch_test_base import PyTorchTestBase
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'mixins'))
from container_engine import ContainerEngineMixin  # noqa: E402


@rfm.simple_test
class PyTorchDdpCeNv(PyTorchTestBase, ContainerEngineMixin):
    descr = ('Check the training throughput using the ContainerEngine and '
             'Alps Extended Image')
    valid_systems = ['+ce +nvgpu']
    maintainers = ['ml-team']
    alps_extended_image = True
    container_image = (
        'jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/'
        'ngc-pytorch:26.02-py3-alps6'
    )
    env_vars = {
        'NCCL_DEBUG': 'Info',
        'SLURM_NETWORK': 'disable_rdzv_get',
    }
    tags = {'production', 'ce'}


@rfm.simple_test
class PyTorchDdpCeNvlarge(PyTorchDdpCeNv):
    num_nodes = parameter([3, 8])


# FIXME: libc comptibility issue on Clariden
# + srun -l --gpus-per-task=1 python cnn_distr.py
# srun: /lib64/libc.so.6: version `GLIBC_2.34' not found
# (required by /opt/cscs/aws-ofi-ccl-plugin/cuda12/libnccl-net.so)
# TODO: build libnccl-net.so plug-in in the test setup phase
@rfm.simple_test
class PyTorchDdpMambaNv(PyTorchTestBase):
    descr = 'Check the training throughput on bare-metal'
    valid_systems = []  #DISABLED TEST, change to ['+nvgpu'] to renable it
    maintainers = ['ml-team']
    time_limit = '30m'
    torch_version = parameter([
         'nccl cuda=12.6', # Latest cu12.6; incompatible driver
    ])
    tags = {'production'}

    @run_after('setup')
    def activate_venv(self):
        self.prerun_cmds = [
            f'set -xe', f'. setup_conda.sh $PWD/forge',
            f'conda update -n base -c conda-forge conda',
            f'conda clean --all',
            f'conda create -p $PWD/forge/envs/rfm {self.torch_version} '
            f'-c nvidia -y',
            f'conda activate $PWD/forge/envs/rfm',
            f'pip install torch torchvision torchaudio --index-url '
            f'https://download.pytorch.org/whl/cu126',
            f'pip install python-hostlist',
            f'. activate_ofi.sh cuda12',
        ]

        self.postrun_cmds = ['rm Miniforge*.sh', 'rm -rf forge']
