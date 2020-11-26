# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuBurnTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100']
        self.descr = 'GPU burn test'
        self.valid_prog_environs = ['PrgEnv-gnu']

        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.modules = ['cuda/10.1.243']
            self.executable_opts = ['-d', '40']
            self.num_gpus_per_node = 8
            gpu_arch = '70'
        elif self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-accel-nvidia60']
            self.executable_opts = ['-d', '20']
        elif self.current_system.name in {'ault'}:
            self.modules = ['cuda']
            self.executable_opts = ['-d', '10']

        self.sourcepath = 'gpu_burn.cu'
        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-lcuda', '-lcublas', '-lnvidia-ml']
        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall('OK', self.stdout)), self.num_tasks_assigned)

        patt = r'GPU\s+\d+\(\S*\): (?P<perf>\S*) GF\/s  (?P<temp>\S*) Celsius'
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(patt, self.stdout, 'perf', float)),
        }

        self.reference = {
            'dom:gpu': {
                'perf': (4115, -0.10, None, 'Gflop/s'),
            },
            'daint:gpu': {
                'perf': (4115, -0.10, None, 'Gflop/s'),
            },
            'arolla:cn': {
                'perf': (5861, -0.10, None, 'Gflop/s'),
            },
            'tsa:cn': {
                'perf': (5861, -0.10, None, 'Gflop/s'),
            },
            'ault:amda100': {
                'perf': (17552, -0.10, None, 'Gflop/s'),
            },
            'ault:amdv100': {
                'perf': (6203, -0.10, None, 'Gflop/s'),
            },
            'ault:intelv100': {
                'perf': (6203, -0.10, None, 'Gflop/s'),
            },
        }

        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.maintainers = ['AJ', 'TM']
        self.tags = {'diagnostic', 'benchmark', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks * self.num_gpus_per_node

    @rfm.run_before('compile')
    def set_gpu_arch(self):
        cs = self.current_system.name
        cp = self.current_partition.fullname
        gpu_arch = None
        if cs in {'dom', 'daint'}:
            gpu_arch = '60'
        elif cs in {'arola', 'tsa'}:
            gpu_arch = '70'
        elif cp in {'ault:amdv100', 'ault:intelv100'}:
            gpu_arch = '70'
        elif cp in {'ault:amda100'}:
            gpu_arch = '80'

        if gpu_arch:
            self.build_system.cxxflags = ['-arch=compute_%s' % gpu_arch,
                                          '-code=sm_%s' % gpu_arch]

    @rfm.run_before('run')
    def set_gpus_per_node(self):
        cs = self.current_system.name
        cp = self.current_partition.fullname
        if cs in {'dom', 'daint'}:
            self.num_gpus_per_node = 1
        elif cs in {'arola', 'tsa'}:
            self.num_gpus_per_node = 8
        elif cp in {'ault:amda100', 'ault:intelv100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdv100'}:
            self.num_gpus_per_node = 2
        else:
            self.num_gpus_per_node = 1
