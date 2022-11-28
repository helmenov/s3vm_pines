# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['s3vm_pines']

package_data = \
{'': ['*'], 's3vm_pines': ['recategolize/*']}

install_requires = \
['SciencePlots>=1.0.9,<2.0.0',
 'indianpines @ git+https://github.com/helmenov/indianpines.git',
 'matplotlib>=3.6.2,<4.0.0',
 'numpy>=1.23.5,<2.0.0',
 'pandas>=1.5.2,<2.0.0',
 'qns3vm @ git+https://github.com/helmenov/qns3vm.git']

setup_kwargs = {
    'name': 's3vm-pines',
    'version': '0.1.7',
    'description': 'A study on Multiclass S3VM for IndianPines',
    'long_description': '# A study on multiclass S3VM for Indian Pines\n\nThis Package provides some tools for a study.\n\n- `recategorize17to10_csv` : csv map to recategorize original 17 categories to 10 categories\n- `train_test_split()` : function to split annotated data into train and test (proportion with respect to the numbers of annotated data is defined `prop_train`)\n- `labeled_unlabeled_test_split()` : function to split annotated data into labeled, unlabeled, and test (unlabeled proportion with respect to the numbers of train data is defined `prop_train_l`)\n- `colored_map()` : plot land cover image colored by category\n\nAnd, experiment examples under [`exp` directory](exp).\n\n## Using\n\n`pip install git+https://www.github.com/helmenov/s3vm_pines`\n\nand, in your python code,\n\n`from s3vm_pines import module`\n\n## functions\n\n### `s3vm_pines.module.train_test_split()`\n\nNote: train dataset is sampled aggromeratively in the map space. it is NOT random sampled.\n\nArgs:\n\n- `prop_train` : (float) train data proportion with respect to the number of annotated data. default is 0.5, it means train:test = 1:1\n- `recategorize_rule` : (str) csv file name which defines recategorize rule. default is `recategorize17to10_csv`. if `None` then it means "not to recategorize": using original 17 categories.\n- `gt_gic` : (bool) read [IndianPines package](https://www.github.com/helmenov/IndianPines). default is True\n\nOutputs: `status, status_name` (tuple)\n\n- `status` : (list[int]) status_number\'s list for each data ID\n- `status_names` : (list[str]) status_label\'s for each status_number.\n    it must be `[\'backgound\', \'test\', \'train\']`\n\n### `s3vm_pines.module.labeled_unlabeled_test_split()`\n\nArgs:\n\n- `prop_train_l` : (float) labeled data proportion with *respect to the number of train data*. (not to the number of annotated data). the proportion with respect to the number of annotated data is `prop_train_l * prop_train`\n- `status` : (list[int]) status_number\'s list for each data ID (provided from `train_test_split`)\n- `unlabeled_type`: (str) `\'from_train\'` or `\'from_spatial\'` or `\'from_annot\'`\n    - `\'from_train\'`: unlabeled data is sampled from train data\n    - `\'from_annot\'`: unlabeled data is sampled from annotated data except for labeled data in random sampling\n    - `\'from_spatialneighbor\'`: unlabeled data is sampled from annotated data except for labeled data in a condition: neighboring with labeled data and the coherency is over `coh_threshold`\n- `coh_thresold`: (float) coherency infmum threshold (>0, <1). high coherency means similar. low coherency means independent.\n- `seed_l`: random seed for defining start index for sampling labeled data from training data\n- `seed_u`: random seed for random sampling unlabeled data from annotated data except for labeled data\n- `recategorize_rule`: you should set same value with that of `train_test_split`\n- `gt_gic`: you should set same value with that of `train_set_split`\n\nOutputs: `l_u_t_status, l_u_t_status_name` (tuple)\n\n- `l_u_t_status` : (list[int]) status_number\'s list for each data ID\n- `l_u_t_status_names` : (list[str]) status_label\'s for each status_number.\n    it must be `[\'background\', \'test\', \'unlabeled\', \'labeled\']`\n\n### `s3vm_pines.module.colored_map()`\n\nplot land-cover colored image with the legend of categories.\n\ntarget data is colored heavily and rest annoted data is colored lightly in background.\n\nArgs:\n\n- `ax`: (`matplotlib.pyplot.figure` object) axes ploted. you should prepare `ax` before you use this function\n- `target`: (`list[int]`) category number\'s list for sample ID\n- `cordinates`: (`list[list[int],list[int]]`) cordinates list for sample ID. 1st column is \'x\'-axis, and 2nd column is \'y\'-axis\n- `recategorize_rule` : you should set same value with that of `train_test_split`\n- `gt_gic`: you should set same value with that of `train_test_split`\n\nOutput: None\n\n\n',
    'author': 'Kotaro SONODA',
    'author_email': 'kotaro1976@gmail.com',
    'maintainer': 'Kotaro SONODA',
    'maintainer_email': 'kotaro1976@gmail.com',
    'url': 'https://github.com/helmenov/s3vm_pines',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.11',
}


setup(**setup_kwargs)
