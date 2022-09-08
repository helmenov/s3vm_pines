
実際に5つの3x3連続領域を指定すると，

```{python}
from s3vm_pines import module as my
from matplotlib import pyplot as plt
import indianpines as IP
import pandas as pd
from importlib import resources
```

```{python}
status, status_name = my.make_traininigSet(Area=4,NumWanted=5,seed=20200810)

recategorize_csv = resources.files('s3vm_pines')/'recategolize'/'recategorize17to10.csv'

IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_csv,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : True,
    }
pines = IP.load(**IP_conf)
```

```{python}

tra_features = pines.features[status==1]
tra_target = pines.target[status==1]
tra_cordinates = pines.cordinates[status==1]
tes_features = pines.features[status==2]
tes_target = pines.target[status==2]
tes_cordinates = pines.cordinates[status==2]
unl_features = pines.features[status==0]
unl_target = pines.target[status==0]
unl_cordinates = pines.cordinates[status==0]
```

```{python}
my.colored_map(tra_target,tra_cordinates)
```

```{python}
my.colored_map(tes_target,tes_cordinates)
```

