# A study on multiclass S3VM for Indian Pines

This Package provides some tools for a study.

- `recategorize17to10_csv` : csv map to recategorize original 17 categories to 10 categories
- `train_test_split()` : function to split annotated data into train and test (proportion with respect to the numbers of annotated data is defined `prop_train`)
- `labeled_unlabeled_test_split()` : function to split annotated data into labeled, unlabeled, and test (unlabeled proportion with respect to the numbers of train data is defined `prop_train_l`)
- `colored_map()` : plot land cover image colored by category

And, experiment examples under [`exp` directory](exp).

## Using

`pip install git+https://www.github.com/helmenov/s3vm_pines`

and, in your python code,

`from s3vm_pines import module`

## functions

### `s3vm_pines.module.train_test_split()`

Note: train dataset is sampled aggromeratively in the map space. it is NOT random sampled.

Args:

- `prop_train` : (float) train data proportion with respect to the number of annotated data. default is 0.5, it means train:test = 1:1
- `recategorize_rule` : (str) csv file name which defines recategorize rule. default is `recategorize17to10_csv`. if `None` then it means "not to recategorize": using original 17 categories.
- `gt_gic` : (bool) read [IndianPines package](https://www.github.com/helmenov/IndianPines). default is True

Outputs: `status, status_name` (tuple)

- `status` : (list[int]) status_number's list for each data ID
- `status_names` : (list[str]) status_label's for each status_number.
    it must be `['backgound', 'test', 'train']`

### `s3vm_pines.module.labeled_unlabeled_test_split()`

Args:

- `prop_train_l` : (float) labeled data proportion with *respect to the number of train data*. (not to the number of annotated data). the proportion with respect to the number of annotated data is `prop_train_l * prop_train`
- `status` : (list[int]) status_number's list for each data ID (provided from `train_test_split`)
- `unlabeled_type`: (str) `'from_train'` or `'from_spatial'` or `'from_annot'`
    - `'from_train'`: unlabeled data is sampled from train data
    - `'from_annot'`: unlabeled data is sampled from annotated data except for labeled data in random sampling
    - `'from_spatialneighbor'`: unlabeled data is sampled from annotated data except for labeled data in a condition: neighboring with labeled data and the coherency is over `coh_threshold`
- `coh_thresold`: (float) coherency infmum threshold (>0, <1). high coherency means similar. low coherency means independent.
- `seed_l`: random seed for defining start index for sampling labeled data from training data
- `seed_u`: random seed for random sampling unlabeled data from annotated data except for labeled data
- `recategorize_rule`: you should set same value with that of `train_test_split`
- `gt_gic`: you should set same value with that of `train_set_split`

Outputs: `l_u_t_status, l_u_t_status_name` (tuple)

- `l_u_t_status` : (list[int]) status_number's list for each data ID
- `l_u_t_status_names` : (list[str]) status_label's for each status_number.
    it must be `['background', 'test', 'unlabeled', 'labeled']`

### `s3vm_pines.module.colored_map()`

plot land-cover colored image with the legend of categories.

target data is colored heavily and rest annoted data is colored lightly in background.

Args:

- `ax`: (`matplotlib.pyplot.figure` object) axes ploted. you should prepare `ax` before you use this function
- `target`: (`list[int]`) category number's list for sample ID
- `cordinates`: (`list[list[int],list[int]]`) cordinates list for sample ID. 1st column is 'x'-axis, and 2nd column is 'y'-axis
- `recategorize_rule` : you should set same value with that of `train_test_split`
- `gt_gic`: you should set same value with that of `train_test_split`

Output: None


