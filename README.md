<h1>Interaction Information Based Automated Feature Engineering (IIFE)</h1>

This is the private version of the repo used for the review process. The public version which should be interacted with is at https://github.com/tomoverman/IIFE.

<h2>How to run</h2>
See requirements.txt for the python packages needed for running IIFE, these can be installed using  pip install -r requirements.txt. See test_iife.py for a reference test script that can be simply run with python test_iife.py. 
Most of this file is loading in data, hyperparameter tuning, finding test scores before and after AutoFE, etc. The line for calling IIFE is on line 185.

<h2>How to test new data</h2>
To test new data, add your data loading file in src/data.py. There are a couple data loading scripts already there that you can copy the structure from. 
Then add this data loading function to your test script (such as test_iife.py).
Furthermore, you will need to also adjust hyperparam_tune.py to include this new data.

<h2>Important IIFE hyperparameters</h2>
See src/iife.py for possible variables to adjust in the function arguments. <b>For most cases the default choices can be used and no adjustment is necessary.</b> The key parameters that may adjust performance are

<b>K</b>: the number of feature pairs to consider each iteration


<b>patience</b>: The stop condition patience, this controls how quickly the algorithm will terminate if there are no longer improvements in cross-validation scores. If this is set larger, then typically more engineered features will be added.

<b>n_int_inf</b>: number of features to consider when calculating interaction information (scales like n choose 2 so be careful when selecting this)

<b>int_inf_subset</b>: size of random subset of samples used for calculating interaction information

<h2>Description of src/ files</h2>

<b>data.py</b>: Data loading functions.

<b>helper.py</b>: All helper functions that are called in iife.py such as evaluating new features, performing normalization and one hot encoding, forming new candidate bivariate functions and unary transformations, etc.

<b>iife.py</b>: The main backbone of the IIFE algorithm which iteratively constructs new features.

<b>knnmci.py</b>: This is the component of IIFE that finds the conditional mutual information as part of the calculation of interaction information. This was copied from the open-source GitHub directory at https://github.com/omesner/knncmi. 
This code was NOT written by us but was used in the IIFE algorithm. 
We copy the code here because the pip install for knncmi no longer works as it is not maintained by the previous authors.
