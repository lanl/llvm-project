# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This package definition is based on the LLVM package in the Spack source tree
# (var/spack/repos/builtin/packages/llvm/package.py) modified to fit the
# newly adopted monorepo layout and for Kitsune's specific needs.


from spack import *
import llnl.util.tty as tty


class Kitsune(CMakePackage):
    """Kitsune is a fork of LLVM that enables optimization within on-node parallel
    constructs by replacing opaque runtime function calls with true parallel
    entities at the LLVM IR level.
    """

    homepage = 'https://github.com/lanl/kitsune'
    url = 'https://github.com/lanl/kitsune/archive/kitsune-0.8.0.tar.gz'
    family = 'compiler'  # Used by lmod
    git = 'https://github.com/lanl/kitsune.git'

    
    # Ninja seems to build LLVM a little faster than Makefiles, but it's not
    # clear how machine-dependent this is.  This would introduce another
    # dependency to the package, which isn't great, and a lot of the developer
    # benefits of Ninja aren't realized when it's being used by an automated
    # system like spack.
    # TODO: Test this a bit.
    # generator = 'Ninja'

    version('0.8.0', tag='kitsune-0.8.0')
    version('develop', branch='release/8.x')


    # Mapping of Spack spec architecture family (lower-case) to the
    # corresponding LLVM target as used in the CMake var LLVM_TARGETS_TO_BUILD.
    target_arch_mapping = {
    #   Spack     : LLVM
        'x86'     : 'X86',
        'x86_64'  : 'X86',
        'arm'     : 'ARM',
        'aarch64' : 'AArch64',
        'sparc'   : 'Sparc',
        'ppc'     : 'PowerPC',
        'power'   : 'PowerPC',
    }

    # Omitting less common targets:
    # BPF, Hexagon, Lanai, Mips, MSP430, SystemZ, WebAssembly, XCore
    available_target_archs = ('AArch64', 'AMDGPU', 'ARM', 'NVPTX', 'PowerPC', 'Sparc', 'X86')


    # *** Project Variants

    # Each project is one that can be enabled in the LLVM_ENABLE_PROJECTS CMake
    # var and has the effect of including projects defined in the top-level of
    # the Kitsune project. This does not necessarily imply they are part of
    # LLVM_ALL_PROJECTS (referenced with -DLLVM_ENABLE_PROJECTS=all) which has
    # been a moving target during the monorepo transition.
    projects = {
        'clang' : {
            'default' : True,
            'description' :
            'Build the LLVM C/C++/Objective-C compiler frontend',
        },
        'clang-tools-extra' : {
            'default' : False,
            'description' :
            'Build extra Clang-based tools (clangd, clang-tidy, etc.)',
        },
        'libcxx' : {
            'default' : False,
            'description' :
            'Build the LLVM C++ standard library',
        },
        'libcxxabi' : {
            'default' : False,
            'description' :
            'Build the LLVM C++ ABI library',
        },
        'libunwind' : {
            'default' : False,
            'description' :
            'Build the libcxxabi libunwind',
        },
        'lldb' : {
            'default' : False,
            'description' :
            'Build the LLVM debugger',
        },
        'compiler-rt' : {
            'default' : False,
            'description' :
            'Build LLVM compiler runtime, including sanitizers',
        },
        'lld' : {
            'default' : False,
            'description' :
            'Build the LLVM linker',
        },
        'polly' : {
            'default' : True,
            'description' :
            'Build the LLVM polyhedral optimization plugin',
        },
        'debuginfo-tests' : {
            'default' : False,
            'description' :
            'Build tests for checking debug info generated by clang',
        },
        'openmp' : {
            'default' : True,
            'description' :
            'Build LLVM\'s libomp',
        }
    }


    # make the spack variants for the projects here
    for project, desc in projects.items():
        variant(project, **desc)



    # *** Other Variants
    
    variant(
        'shared_libs',
        description='Build all components as shared libraries, faster, '
        'less memory to build, less stable',
        default=False
    )

    variant(
        'link_dylib',
        description='Build and link the libLLVM shared library rather '
        'than static',
        default=False
    )

    variant(
        'targets',
        description='Targets to build support for, default: '
        '<current arch>, NVPTX, AMDGPU',
        values=
        disjoint_sets(('default', 'all'), available_target_archs)
        .prohibit_empty_set()
        .with_default('default')
        .with_non_feature_values('default', 'all')
    )

    variant(
        'tapir_runtimes',
        description='Parallel runtime(s) to enable for -ftapir=<runtime>',
        values=disjoint_sets(('all',), ('qthreads', 'realm', 'openmp'))
        .with_default('all')
        .with_non_feature_values('all', 'none')
    )

    variant('kokkos',
            description='Enable lowering of Kokkos constructs',
            default=True
            )

    # NOTE: The debug version of LLVM is an order of magnitude larger than
    # the release version, and may take up 20-30 GB of space. If you want
    # to save space, build with `build_type=Release`.
    variant(
        'build_type',
        description='CMake build type',
        default='Release',
        values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')
    )

    variant(
        'python',
        description='Install python bindings',
        default=False,
    )

    variant(
        'gold',        
        description='Add support for LTO with the gold linker plugin',
        default=True
    )


    # *** Build Dependencies

    # NOTE: if libclc project is added, need to require 3.9.2+ here
    depends_on('cmake@3.4.3:', type='build')

    # Even if python bindings aren't being created, we still need it to build
    depends_on('python', when='~python', type='build')


    # *** Runtime Dependencies

    depends_on('py-six', when='+python +lldb')

    depends_on('binutils+gold', when='+gold')

    depends_on('ncurses', when='+lldb')
    depends_on('swig',    when='+lldb')
    depends_on('libedit', when='+lldb')
    
    # openmp uses Data::Dumper in openmp/runtime/tools/lib/tools.pm
    depends_on('perl-data-dumper', type='build')

    # NOTE: this creates the python dependency as well as ensuring python
    # bindings are appropriately linked into the python tree
    extends('python', when='+python')

    # NOTE: Simply requiring {kokkos,legion,qthreads} should be sufficient to
    # enable the respective Kitsune functionality. No need to add CMake defines
    # in cmake_args()
    depends_on('kokkos', when='+kokkos')

    depends_on('legion', when='tapir_runtimes=realm')

    depends_on('qthreads+hwloc', when='tapir_runtimes=qthreads')

    # NOTE: qthreads backend requires linking against libnuma, which is pulled
    # in by the hwloc dependency.  Not sure if it's worth making the Spack
    # numactl package an explicit dependency here or not.
    #
    # depends_on('numactl', when='tapir_runtimes=qthreads')

    # Need to specify these as well, unfortunately.  There is not yet any way to
    # make tapir_runtimes=all implicitly the same as
    # tapir_runtimes=qthreads,realm,openmp
    depends_on('legion', when='tapir_runtimes=all')
    depends_on('qthreads+hwloc', when='tapir_runtimes=all')



    # *** Conflicts

    # it would be nice if '+clang' could be specified as a dependency for lldb
    # and the extra tools, but that's not possible with spack, so this is the
    # next best thing
    conflicts('+lldb', when='~clang')
    conflicts('+clang-tools-extra', when='~clang')

    # Not really sure this is useful.
    # At least it guarantees that the LLVM libomp will be available as a runtime.
    conflicts('~openmp', when='tapir_runtimes=omp')



    # *** Methods

    @run_before('cmake')
    def check_darwin_lldb_codesign_requirement(self):
        """Ensure codesign identity is available for lldb on mac
        """

        if not self.spec.satisfies('+lldb platform=darwin'):
            return
        codesign = which('codesign')
        mkdir('tmp')
        llvm_check_file = join_path('tmp', 'llvm_check')
        copy('/usr/bin/false', llvm_check_file)

        try:
            codesign('-f', '-s', 'lldb_codesign', '--dryrun',
                     llvm_check_file)

        except ProcessError:
            explanation = ('The "lldb_codesign" identity must be available'
                           ' to build LLVM with LLDB. See https://llvm.org/'
                           'svn/llvm-project/lldb/trunk/docs/code-signing'
                           '.txt for details on how to create this identity.')
            raise RuntimeError(explanation)

    
    def setup_environment(self, spack_env, run_env):
        # set the appropriate c++11 flag in the build environment for whatever
        # compiler is being used
        spack_env.append_flags('CXXFLAGS', self.compiler.cxx11_flag)

        # environment vars set in the modulefile
        if '+clang' in self.spec:
            run_env.set('CC', join_path(self.spec.prefix.bin, 'clang'))
            run_env.set('CXX', join_path(self.spec.prefix.bin, 'clang++'))


    # With the new LLVM monorepo, CMakeLists.txt lives in the llvm subdirectory.
    @property
    def root_cmakelists_dir(self):
        """The relative path to the directory containing CMakeLists.txt

        This path is relative to the root of the extracted tarball,
        not to the ``build_directory``. Defaults to the current directory.

        :return: directory containing CMakeLists.txt
        """
        return 'llvm'



    def cmake_args(self):
        spec = self.spec
        cmake_args = [
            '-DLLVM_REQUIRES_RTTI:BOOL=ON',
            '-DLLVM_ENABLE_RTTI:BOOL=ON',
            '-DLLVM_ENABLE_EH:BOOL=ON',
            '-DCLANG_DEFAULT_OPENMP_RUNTIME:STRING=libomp',
            '-DPYTHON_EXECUTABLE:PATH={0}'.format(spec['python'].command.path),
        ]

        # TODO: Instead of unconditionally disabling CUDA, add a "cuda" variant
        #       (see TODO in llvm spack package), and set the paths if enabled.
        cmake_args.extend([
            '-DCUDA_TOOLKIT_ROOT_DIR:PATH=IGNORE',
            '-DCUDA_SDK_ROOT_DIR:PATH=IGNORE',
            '-DCUDA_NVCC_EXECUTABLE:FILEPATH=IGNORE',
            '-DLIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES:STRING=IGNORE'])


        if '+shared_libs' in spec:
            cmake_args.append('-DBUILD_SHARED_LIBS:BOOL=ON')

        if '+link_dylib' in spec:
            cmake_args.append('-DLLVM_LINK_LLVM_DYLIB:BOOL=ON')

        if '+libcxx' in spec:
            cmake_args.append('-DCLANG_DEFAULT_CXX_STDLIB:STRING=libc++')

        if '+polly' in spec:
            cmake_args.append('-DLLVM_LINK_POLLY_INTO_TOOLS:BOOL=ON')

        if '+gold' in spec:
            cmake_args.append('-DLLVM_BINUTILS_INCDIR:PATH=' +
                              spec['binutils'].prefix.include)

        if '+python' in spec and '+lldb' in spec:
            cmake_args.append('-DLLDB_USE_SYSTEM_SIX:BOOL=TRUE')


        enable_projects=[]
        for project in Kitsune.projects:
            if '+{}'.format(project) in spec:
                enable_projects.append(project)
        
        cmake_args.append(
            '-DLLVM_ENABLE_PROJECTS:STRING={}'.format(
                ';'.join(enable_projects)))

        
        # all is default in CMake config, so only mess with args if something
        # else is specified
        if 'targets=all' not in spec:
            targets = []
            if 'targets=default' in spec:
                targets.extend(['NVPTX', 'AMDGPU'])
                targets.append(Kitsune.target_arch_mapping.get(str(spec.target.family)))
            else:
                for tgt in available_target_archs:
                    if 'targets={}'.format(tgt) in spec:
                        targets.append(tgt)

            cmake_args.append(
                '-DLLVM_TARGETS_TO_BUILD:STRING=' + ';'.join(targets))
            

        if spec.satisfies('platform=linux'):
            # set the RPATH to the install path at build time (rather than
            # relinking at install time)
            cmake_args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON')
        return cmake_args

    @run_before('build')
    def pre_install(self):
        with working_dir(self.build_directory):
            # When building shared libraries these need to be installed first
            make('install-LLVMTableGen')
            make('install-LLVMDemangle')
            make('install-LLVMSupport')

    @run_after('install')
    def post_install(self):
        if '+clang' in self.spec and '+python' in self.spec:
            install_tree(
                'tools/clang/bindings/python/clang',
                join_path(site_packages_dir, 'clang'))

        with working_dir(self.build_directory):
            install_tree('bin', join_path(self.prefix, 'libexec', 'llvm'))
