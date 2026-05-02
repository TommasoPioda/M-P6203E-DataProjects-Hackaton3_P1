from lime.lime_tabular import LimeTabularExplainer
import shap
from IPython.display import display, HTML

def lime_explainer(X_train, X_test, y_test, model):
    """
    Based on train and test data, it use the LIME to
    explain locally a valid and invalid reference.
    
    Args:
        X_train (pd.DataFrame): Dataframe of the train set
        
        X_test (pd.DataFrame): Dataframe of the test set
        
        y_test (pd.Series): Series containing target values of the test set
        
        model: Model trained over the data
        
        
    Plot a valid and invalid reference, explaining locally
    which features push the model to take such decisions.       
    """

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
        )

    valid_ref = X_test[y_test==1].iloc[0].values
    invalid_ref = X_test[y_test==0].iloc[0].values

    #explain valid reference
    print("Valid reference:")
    exp_val = explainer.explain_instance(
        valid_ref,
        model.predict_proba
        )

    display(HTML(exp_val.as_html()))

    #explain valid reference
    print("\nInvalid reference")
    exp_inval = explainer.explain_instance(
        invalid_ref,
        model.predict_proba
        )

    display(HTML(exp_inval.as_html()))

def shap_tree_explainer(X_test, model):
    """
    Based on test data, it use the SHAP to
    explain a tree model globally, showing
    which features influence positively and negatively
    the decision.
    
    Args:
        X_test (pd.DataFrame): Dataframe of the test set
        
        model (TreeModel): Tree model trained over the data
        
        
    Plot a summary and a bar plot over the test set    
    """

    
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_test)

    # summary plot
    shap.summary_plot(shap_val, X_test)

    # bar plot
    shap.summary_plot(shap_val, X_test, plot_type="bar")