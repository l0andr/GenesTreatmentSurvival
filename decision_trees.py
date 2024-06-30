import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import  numpy as np
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#from matplotlib.backends.backend_pdf import PdfPages
import warnings
import argparse
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from utility_functions import prepare_columns_for_model
from sklearn.preprocessing import OneHotEncoder
from odtlearn.flow_oct import FlowOCT, BendersOCT
from odtlearn.flow_opt import FlowOPT_IPW
from odtlearn.utils.binarize import binarize

def decision_tree_hyper_parameter_grid_search(X, data_y, min_weight_fraction_leaf, ccp_alpha, min_samples_leaf, max_depth,
                                         criterion='entropy',optimized_parameter="min_weight_fraction_leaf",min_value=0.01,max_value=0.1,npoint=10):
    td_acc = []
    cv_acc = []
    pscale = np.linspace(min_value,max_value,npoint)
    for p in pscale:
        if optimized_parameter == "min_weight_fraction_leaf":
            min_weight_fraction_leaf = p
        if optimized_parameter == "ccp_alpha":
            ccp_alpha = p
        if optimized_parameter == "min_samples_leaf":
            min_samples_leaf = int(p)
        if optimized_parameter == "max_depth":
            max_depth = int(p)
        model = DecisionTreeClassifier(min_weight_fraction_leaf=min_weight_fraction_leaf, ccp_alpha=ccp_alpha,
                                                   min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                                   criterion=criterion,
                                                   random_state=2).fit(X, data_y)
        td_acc.append(model.score(X, data_y))
        cv_acc.append(cross_val_score(model, X, data_y, cv=5).mean())
    return td_acc, cv_acc,pscale

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Transform data from initial csv to csv suitable for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("--input_delimiter", help="Delimiter for input file", type=str, default=",")
    parser.add_argument("-ycolumn", help="Feature that should be predicted", type=str, required=True)
    parser.add_argument("--xcolumns", help="Features that should be used for split branch", type=str, default="")
    parser.add_argument("-output_pdf", help="Output PDF file", type=str, default="")
    parser.add_argument("--sort_columns", help="Columns for pre-sort data before processing", type=str, default="")
    parser.add_argument("--unique_record", help="List of columns to identify unique records", type=str, default="")
    parser.add_argument("--verbose", help="Verbose level", type=int, default=2)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--min_weight_fraction_leaf", help="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node", type=float, default=None)
    parser.add_argument("--min_samples_leaf", help="The minimum number of samples required to be at a leaf node", type=int, default=None)
    parser.add_argument("--max_depth", help="The maximum depth of the tree", type=int, default=None)
    parser.add_argument("--ccp_alpha", help="Complexity parameter used for Minimal Cost-Complexity Pruning", type=float, default=None)
    parser.add_argument("--min_impurity_decrease", help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value", type=float, default=None)
    parser.add_argument("--max_features", help="The number of features to consider when looking for the best split", type=int, default=None)
    parser.add_argument("--max_leaf_nodes", help="Grow a tree with max_leaf_nodes in best-first fashion", type=int, default=None)
    parser.add_argument("--min_samples_split", help="The minimum number of samples required to split an internal node", type=int, default=None)
    parser.add_argument("--steps_of_optimization", help="Number of steps for optimization", type=int, default=20)
    parser.add_argument("--criteria", help="The function to measure the quality of a split", type=str, choices=['entropy','gini'], default='gini')
    parser.add_argument("--class_names", help="List of class names", type=str, default="")

    args = parser.parse_args()
    input_csv = args.input_csv
    input_delimiter = args.input_delimiter
    warnings.simplefilter(action='ignore', category=FutureWarning)
    tdf_tree = pd.read_csv(input_csv, delimiter=input_delimiter)
    #tdf_tree = tdf_tree[tdf_tree['tnum']>1]
    #select only treatment_type == 2
    #tdf_tree = tdf_tree[tdf_tree['treatment_type'] == 2]
    if args.sort_columns:
        tdf_tree.sort_values(by=args.sort_columns.split(','), inplace=True)
    if args.unique_record:
        tdf_tree = tdf_tree.drop_duplicates(subset=args.unique_record.split(','), keep='first')
    if args.xcolumns:
        tdf_tree = tdf_tree[args.xcolumns.split(',') + [args.ycolumn]]
    #supress FutureWarning
    criterium = args.criteria
    #kepp records with treatment_type
    tdf_tree = tdf_tree[tdf_tree['treatment_type'] == 2]

    for col in tdf_tree.columns:
        try:
            if tdf_tree[col].nunique() == 2:
                list_of_values = sorted(tdf_tree[col].unique(), reverse=True)
                if args.verbose > 1:
                    print(f"Column {col} has 2 unique values, True is  {list_of_values[0]}")
                #check if str is in list of values
                tdf_tree[col] =tdf_tree[col].apply(lambda x: x == list_of_values[0])
                tdf_tree[col] = tdf_tree[col].astype(int)
            #if column not numeric, convert it to integer renumber the values
            if tdf_tree[col].dtype == 'object':
                # print unique values and cat codes
                if args.verbose > 1:
                    print(f"Column {col} has {tdf_tree[col].nunique()} unique values")
                    print(tdf_tree[col].value_counts())
                tdf_tree[col] = tdf_tree[col].astype('category').cat.codes
            #if column has missing values, fill them with median
            if tdf_tree[col].isnull().sum() > 0:
                tdf_tree[col] = tdf_tree[col].fillna(tdf_tree[col].median())
        except Exception as e:
            import traceback
            print(f"Error in column {col}: {e}. Column will be dropped")
            traceback.print_exc()
            tdf_tree.drop(columns=[col],inplace=True)


    # data_y = tdf[['status', 'disease_free_time']].to_records(index=False)
    print(args.ycolumn)
    data_y = tdf_tree[args.ycolumn]
    # X should contain all co,umns from tdf started from gene_
    X = tdf_tree.drop(columns=[args.ycolumn])
    # convert all columns of X to categorical
    #treat = np.array(X['treatment_type'].astype(int).tolist()) -1
    #ipw = np.zeros(treat.shape[0])+1
    #select all columns that contain gene_ perfix as well sex,smokig


    #list_of_columns_for_p_tree = X.columns.str.contains('gene_', case=False)
    #X = X.loc[:,list_of_columns_for_p_tree]
    '''
    optimized_parameter = "min_weight_fraction_leaf"
    td,cv,ps = decision_tree_hyper_parameter_grid_search(X, data_y, args.min_weight_fraction_leaf, args.ccp_alpha, args.min_samples_leaf,args.max_depth,
                                              criterion='entropy', optimized_parameter=optimized_parameter,
                                              min_value=0.001, max_value=0.2, npoint=40)

    #find ps coresponding maximum cv
    max_cv = np.max(cv)
    max_cv_index = cv.index(max_cv)
    print(f"Optimal value of {optimized_parameter} is {ps[max_cv_index]}")
    plt.figure()
    plt.plot(ps,td,label='Accuracy on train data')
    plt.plot(ps,cv,label='Cross validated accuracy')
    plt.legend()
    plt.show()
    '''

    params_dict = {'max_depth':args.max_depth,
     'min_samples_leaf':args.min_samples_leaf,
     'min_weight_fraction_leaf':args.min_weight_fraction_leaf,
     'ccp_alpha':args.ccp_alpha,
     'min_impurity_decrease':args.min_impurity_decrease,
     'max_features':args.max_features,
     'max_leaf_nodes':args.max_leaf_nodes,
     'min_samples_split':args.min_samples_split}
    list_of_int_parameters = ['max_depth','min_samples_leaf','max_features','max_leaf_nodes','min_samples_split']
    list_of_real_parameters = ['min_weight_fraction_leaf','ccp_alpha','min_impurity_decrease']
    space = []
    list_of_parameters_for_optimization = []
    dictionary_of_fixed_parameters = {}
    for key,value in params_dict.items():
        if value is None:
            if key in list_of_int_parameters:
                space.append(Integer(2, 30, name=key))
                if args.verbose > 1:
                    print(f"Parameter {key} subjected for optimization in range 2-30")
            elif key in list_of_real_parameters:
                space.append(Real(0.000001, 0.5, name=key))
                if args.verbose > 1:
                    print(f"Parameter {key} subjected for optimization in range 0.000001-0.5")
            else:
                raise Exception(f"Unknown parameter type for {key}")
            list_of_parameters_for_optimization.append(key)
        else:
            dictionary_of_fixed_parameters[key] = value

    @use_named_args(space)
    def objective1(**params):
        model = DecisionTreeClassifier(**params,**dictionary_of_fixed_parameters,
                                       criterion=criterium,random_state=0).fit(X, data_y)
        return -model.score(X, data_y) # -np.mean(cross_val_score(model, X, data_y, cv=10))

    @use_named_args(space)
    def objective2(**params):
        model = DecisionTreeClassifier(**params,**dictionary_of_fixed_parameters,
                                       criterion=criterium,random_state=0).fit(X, data_y)
        return -(cross_val_score(model, X, data_y, cv=5).mean())

    from skopt import gp_minimize
    res_gp = gp_minimize(objective1, space, n_calls=args.steps_of_optimization, random_state=0,verbose=True,n_jobs=-1)
    print("Results of optimization:")
    dictionary_of_optimised_parameters = {}
    for i in range(0,len(list_of_parameters_for_optimization)):
        if args.verbose > 1:
            print(f"{list_of_parameters_for_optimization[i]}:{res_gp.x[i]}")
        dictionary_of_optimised_parameters[list_of_parameters_for_optimization[i]] = res_gp.x[i]
    from skopt.plots import plot_convergence
    joint_params_dict = {**dictionary_of_optimised_parameters,**dictionary_of_fixed_parameters}
    model = DecisionTreeClassifier(**joint_params_dict,
                                   criterion=criterium,
                                   random_state=0).fit(X, data_y)

    #permutation importance
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, data_y,n_repeats = 30,random_state = 0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<8} "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
    predictions = model.predict(X)
    print(f"In-sample accuracy is {np.sum(predictions == data_y) / data_y.shape[0]}")
    warnings.resetwarnings()
    output_name = f"{args.output_pdf}_{joint_params_dict['ccp_alpha']:.4f}_{joint_params_dict['min_samples_leaf']}_{joint_params_dict['max_depth']}_{joint_params_dict['min_weight_fraction_leaf']:.2f}.pdf"
    pp = PdfPages(output_name)
    class_names = args.class_names.split(',')
    if len(class_names) == 0:
        class_names = None

    for i in range(0,3):
        plt.figure(figsize=(20, 10))
        model = DecisionTreeClassifier(**joint_params_dict,
                                       criterion=criterium,
                                       random_state=i).fit(X, data_y)
        print(f"Model accuracy on train data:{model.score(X, data_y)}")
        scores = cross_val_score(model, X, data_y, cv=5)

        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        title = f"Decision tree with ccp_alpha={joint_params_dict['ccp_alpha']}, min_samples_leaf={joint_params_dict['min_samples_leaf']}, max_depth={joint_params_dict['max_depth']}, " \
                f"min_weight_fraction_leaf={joint_params_dict['min_weight_fraction_leaf']}"
        title += f"\nModel score on train data:{model.score(X, data_y)}\n"
        title += f"Cross validated accuracy {scores.mean():.2f}  with a standard deviation of {scores.std():.2f}"

        plot_tree(model, feature_names=X.columns, filled=True, fontsize=10,class_names=class_names)
        plt.tight_layout()
        plt.title(title)
        pp.savefig()
    plt.figure(figsize=(20, 10))
    plot_convergence(res_gp)
    pp.savefig()
    pp.close()
    warnings.resetwarnings()

    #import matplotlib
    #matplotlib.use('agg')
    '''
    stcl = FlowOPT_IPW(depth=4, solver="CBC", time_limit=20,num_threads=8)

    stcl.fit(X, treat,data_y,ipw=ipw)
    plt.figure(figsize=(20, 10))
    stcl.plot_tree(filled=True)
    stcl.print_tree()
    predictions = stcl.predict(X)
    print(f"In-sample accuracy is {np.sum(predictions == data_y) / data_y.shape[0]}")
    '''
    if args.show:
        plt.show()

