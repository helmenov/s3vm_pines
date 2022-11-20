# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['s3vm_pines']

package_data = \
{'': ['*'], 's3vm_pines': ['recategolize/*']}

install_requires = \
['SciencePlots>=1.0.9,<2.0.0',
 'indianpines @ git+https://github.com/helmenov/indianpines.git']

setup_kwargs = {
    'name': 's3vm-pines',
    'version': '0.1.2',
    'description': '',
    'long_description': '\n実際に5つの3x3連続領域を指定すると，\n\n```{python}\nfrom s3vm_pines import module as my\nfrom matplotlib import pyplot as plt\nimport indianpines as IP\nimport pandas as pd\nfrom importlib import resources\n```\n\n```{python}\nstatus, status_name = my.make_traininigSet(Area=4,NumWanted=5,seed=20200810)\n\nrecategorize_csv = resources.files(\'s3vm_pines\')/\'recategolize\'/\'recategorize17to10.csv\'\n\nIP_conf = {\n        "pca": 5,\n        "include_background": True,\n        "recategorize_rule" : recategorize_csv,\n        "exclude_WaterAbsorptionChannels" : True,\n        "gt_gic" : True,\n    }\npines = IP.load(**IP_conf)\n```\n\n```{python}\n\ntra_features = pines.features[status==1]\ntra_target = pines.target[status==1]\ntra_cordinates = pines.cordinates[status==1]\ntes_features = pines.features[status==2]\ntes_target = pines.target[status==2]\ntes_cordinates = pines.cordinates[status==2]\nunl_features = pines.features[status==0]\nunl_target = pines.target[status==0]\nunl_cordinates = pines.cordinates[status==0]\n```\n\n```{python}\nmy.colored_map(tra_target,tra_cordinates)\n```\n\n```{python}\nmy.colored_map(tes_target,tes_cordinates)\n```\n\n',
    'author': 'Kotaro SONODA',
    'author_email': 'kotaro1976@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.11',
}


setup(**setup_kwargs)
