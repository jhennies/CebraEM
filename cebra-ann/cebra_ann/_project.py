
import os
import numpy as np
import json


mem_defaults = dict(
    shape=None,  # Use the full raw data shape
    halo=16,
    padding='zeros',
    batch_size=64,
    sigma=0.0,
    qnorm_low=0.0,
    qnorm_high=1.0
)


sv_defaults = dict(
    shape=None,
    halo=32,
    padding='zeros',
    threshold=0.05,
    min_membrane_size=1,
    sigma_dt=1.0,
    min_segment_size=48
)


defaults = dict(
    path=None,
    raw=None,
    mem=None,
    sv=None,
    raw_locked=False,
    mem_locked=False,
    sv_locked=False,
    pre_merge=None,
    pre_merge_beta=None,
    beta=0.5,
    instances=None,
    instances_brush_size=None,
    brush_size=5,
    instance_seg_running=False,
    semantics=dict(),  # A dict of the form {semantic_name: filepath}
    semantic_names=[],
    semantic_types=dict(),
    mem_params=mem_defaults,
    sv_params=sv_defaults
)


class AnnProject:

    def __init__(self, path):

        self.valid = validate_project(path)

        # --- Properties that will be saved when calling self.save() ---
        self.path = path
        self.raw = defaults['raw']
        self.mem = defaults['mem']
        self.sv = defaults['sv']
        self.raw_locked = defaults['raw_locked']
        self.mem_locked = defaults['mem_locked']
        self.sv_locked = defaults['sv_locked']
        self.pre_merge = defaults['pre_merge']
        self.pre_merge_beta = defaults['pre_merge_beta']
        self.beta = defaults['beta']
        self.instances = defaults['instances']
        self.instances_brush_size = defaults['instances_brush_size']
        self.brush_size = defaults['brush_size']
        self.instance_seg_running = defaults['instance_seg_running']
        self.semantics = defaults['semantics']
        self.semantic_names = defaults['semantic_names']
        self.semantic_types = defaults['semantic_types']
        self.mem_params = defaults['mem_params']
        self.sv_params = defaults['sv_params']

        # --- Properties that are only used as long as the session is alive ---
        # Flags which state if one of the layers were altered
        self.raw_touched = False
        self.mem_touched = False
        self.sv_touched = False
        self.pre_merge_touched = False
        self.instances_touched = False
        self.semantics_touched = dict()

        self.load()

    def _has_project_file(self):
        return os.path.exists(self.get_project_json())

    def get_project_json(self):
        if self.path is None:
            return None
        else:
            return os.path.join(self.path, 'project.json')

    def _load_from_dict(self, project_info, ignore_path=False):
        if not ignore_path:
            self.path = project_info['path']
        if 'raw' in project_info:
            self.raw = project_info['raw']
        if 'mem' in project_info:
            self.mem = project_info['mem']
        if 'sv' in project_info:
            self.sv = project_info['sv']
        if 'raw_locked' in project_info:
            self.raw_locked = project_info['raw_locked']
        if 'mem_locked' in project_info:
            self.mem_locked = project_info['mem_locked']
        if 'sv_locked' in project_info:
            self.sv_locked = project_info['sv_locked']
        if 'pre_merge' in project_info:
            self.pre_merge = project_info['pre_merge']
        if 'beta' in project_info:
            self.beta = project_info['beta']
        if 'pre_merge_beta' in project_info:
            self.pre_merge_beta = project_info['pre_merge_beta']
        if 'brush_size' in project_info:
            self.brush_size = project_info['brush_size']
        if 'instances' in project_info:
            self.instances = project_info['instances']
        if 'instances_brush_size' in project_info:
            self.instances_brush_size = project_info['instances_brush_size']
        if 'instance_seg_running' in project_info:
            self.instance_seg_running = project_info['instance_seg_running']
        if 'semantics' in project_info:
            self.semantics = project_info['semantics']
        if 'semantic_names' in project_info:
            self.semantic_names = project_info['semantic_names']
            for name in self.semantic_names:
                self.semantics_touched[name] = False
        if 'semantic_types' in project_info:
            self.semantic_types = project_info['semantic_types']
        if 'mem_params' in project_info:
            self.mem_params = project_info['mem_params']
        if 'sv_params' in project_info:
            self.sv_params = project_info['sv_params']

    def _load_from_proj_file(self):
        fp = self.get_project_json()
        assert fp is not None, 'No project.json found!'
        with open(fp, mode='r') as f:
            self._load_from_dict(json.load(f))

    def _load_from_scratch(self):
        # Load the project information
        t_defaults = defaults.copy()
        if self.raw is None:
            if os.path.exists(os.path.join(self.path, 'raw.h5')):
                t_defaults['raw'] = self.default_raw()
        if self.mem is None:
            if os.path.exists(os.path.join(self.path, 'mem.h5')):
                t_defaults['mem'] = self.default_mem()
        if self.sv is None:
            if os.path.exists(os.path.join(self.path, 'sv.h5')):
                t_defaults['sv'] = self.default_sv()
        self._load_from_dict(t_defaults, ignore_path=True)

    def load(self):
        if self.valid:
            # Load the project information
            if self._has_project_file():
                self._load_from_proj_file()
            else:
                self._load_from_scratch()

    def _generate_project_info_dict(self):
        return dict(
            path=self.path,
            raw=self.raw,
            mem=self.mem,
            sv=self.sv,
            raw_locked=self.raw_locked,
            mem_locked=self.mem_locked,
            sv_locked=self.sv_locked,
            pre_merge=self.pre_merge,
            pre_merge_beta=self.pre_merge_beta,
            beta=self.beta,
            instances=self.instances,
            instances_brush_size=self.instances_brush_size,
            brush_size=self.brush_size,
            instance_seg_running=self.instance_seg_running,
            semantics=self.semantics,
            semantic_names=self.semantic_names,
            semantic_types=self.semantic_types,
            mem_params=self.mem_params,
            sv_params=self.sv_params
        )

    def save(self):
        with open(self.get_project_json(), mode='w') as f:
            json.dump(self._generate_project_info_dict(), f, indent=2)

    def set_raw(self, fp=None):
        if fp is None:
            self.raw = self.default_raw()
            self.raw_touched = True
        else:
            path, file = os.path.split(fp)
            self.raw = os.path.join('{project}', file) if path == self.path else fp

    def set_mem(self, fp=None):
        if fp is None:
            self.mem = self.default_mem()
            self.mem_touched = True
        else:
            path, file = os.path.split(fp)
            self.mem = os.path.join('{project}', file) if path == self.path else fp

    def set_sv(self, fp=None):
        if fp is None:
            self.sv = self.default_sv()
            self.sv_touched = True
        else:
            path, file = os.path.split(fp)
            self.sv = os.path.join('{project}', file) if path == self.path else fp

    def set_pre_merge(self, beta):
        self.pre_merge = self.default_pre_merge()
        self.pre_merge_beta = beta
        self.pre_merge_touched = True

    def set_instances(self, brush_size):
        self.instances = self.default_instances()
        self.instances_brush_size = brush_size
        self.instances_touched = True

    def set_semantic(self, name, type):
        self.semantics[name] = self.default_semantic(name)
        self.semantic_names.append(name)
        self.semantic_types[name] = type
        self.semantics_touched[name] = True

    def get_absolute_path(self, rel_path=None, name=None):
        if name is None:
            return rel_path.format(project=self.path)
        else:
            if name == 'raw':
                return self.raw.format(project=self.path)
            elif name == 'mem':
                return self.mem.format(project=self.path)
            elif name == 'sv':
                return self.sv.format(project=self.path)
            elif name == 'pre_merge':
                return self.pre_merge.format(project=self.path)
            elif name == 'instances':
                return self.instances.format(project=self.path)
            elif name[:10] == 'semantics_':
                return self.semantics[name].format(project=self.path)

    def get_semantic_names_and_types(self, names_and_type_str=None):
        """
        Returns a list of strings like such: ["semantics_mito (multi)", "semantics_er (single)", ...]
        """
        if names_and_type_str:
            # Pull a names and types string apart
            res = names_and_type_str.split(' ')
            res[1] = res[1][1: -1]
            return res
        else:
            # Combine names and types
            return(
                [f'{name} ({self.semantic_types[name]})' for name in self.semantic_names]
            )

    def get_all_active_layers(self):
        all_layers = []
        if self.raw is not None:
            all_layers.append('raw')
        if self.mem is not None:
            all_layers.append('mem')
        if self.sv is not None:
            all_layers.append('sv')
        if self.pre_merge is not None:
            all_layers.append('pre_merge')
        if self.instances is not None:
            all_layers.append('instances')
        if len(self.semantics) > 0:
            for sem in self.semantic_names:
                all_layers.append(sem)
        return all_layers

    def set_back(self, step):

        def instances():
            self.instances = None
            self.instance_seg_running = False
            self.instances_touched = False
            self.semantics = defaults['semantics']
            self.semantic_names = defaults['semantic_names']
            self.semantic_types = defaults['semantic_types']
            self.semantics_touched = dict()

        def pre_merge():
            self.pre_merge = None
            self.pre_merge_touched = False
            instances()

        def sv():
            self.sv = None
            self.sv_touched = False
            pre_merge()

        def mem():
            self.mem = None
            self.mem_touched = False
            sv()

        def raw():
            self.raw = None
            self.raw_touched = False
            mem()

        if step == 'sv':
            sv()
        elif step == 'mem':
            mem()
        elif step == 'raw':
            raw()
        elif step == 'pre_merge':
            pre_merge()
        elif step == 'instances':
            instances()
        else:
            raise NotImplementedError(f'step = {step} not implemented!')

    # A collection of static functionalities that are loosely linked to the project and should only be called once a
    # project is loaded
    @staticmethod
    def get_translations(shapes: dict):
        """
        Computes translations for an arbitrary amount of maps, assuming all these maps have the same central point while
        having different shapes.

        :param shapes: dictionary of the form {'raw': raw, 'mem': mem, ...}
        :return: translations in the form {'raw': translation_raw, 'mem': translation_mem, ...}
        """
        translations = dict()
        shp_max = np.max(list(shapes.values()), axis=0)
        print(f'shp_max = {shp_max}')
        for k, v in shapes.items():
            translations[k] = (shp_max - np.array(v)) / 2

        return translations

    @staticmethod
    def default_raw():
        return os.path.join('{project}', 'raw.h5')

    @staticmethod
    def default_mem():
        return os.path.join('{project}', 'mem.h5')

    @staticmethod
    def default_sv():
        return os.path.join('{project}', 'sv.h5')

    @staticmethod
    def default_pre_merge():
        return os.path.join('{project}', 'pre_merge.h5')

    @staticmethod
    def default_instances():
        return os.path.join('{project}', 'instances.h5')

    @staticmethod
    def default_semantic(name):
        assert name[:10] == 'semantics_' and len(name) > 10
        return os.path.join('{project}', f'{name[10:]}.h5')


def validate_project_json(fp):
    # TODO
    return True


def validate_project(path, create=False, empty=False):

    if not os.path.exists(path):
        if create:
            os.mkdir(path)
            return True
        else:
            return False
    else:
        if len(os.listdir(path)) == 0:
            return True
        else:
            if empty:
                return False
            elif os.path.exists(os.path.join(path, 'project.json')):
                if validate_project_json(os.path.exists(os.path.join(path, 'project.json'))):
                    return True
                else:
                    return False
            else:
                if os.path.exists(os.path.join(path, 'raw.h5')):
                    return True
                else:
                    return False
