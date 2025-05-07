import pandas as pd

def load_and_label_raw_data(filepath: str) -> pd.DataFrame:
    columns = [
        'account_status', 'duration_mon', 'credit_history', 'purpose', 'credit_amount', 'savings',
        'employment_yr', 'installment_rate', 'personal_status_sex', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installment_plans', 'housing',
        'existing_credits', 'job', 'num_liable_people', 'telephone', 'foreign_worker', 'target'
    ]
    df = pd.read_csv(filepath, delimiter=' ', header=None)
    df.columns = columns
    df['target'] = df['target'].map({1: 1, 2: 0})
    return df

def replace_categorical_labels(df):
    account_status_map = {
        "A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM / salary assignment", "A14": "no checking account"
    }
    credit_history_map = {
        "A30": "no credits / all paid back", "A31": "all credits paid", "A32": "existing credits paid", 
        "A33": "delayed payments", "A34": "critical account"
    }
    purpose_map = {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture", "A43": "radio/TV", "A44": "appliances",
        "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "other"
    }
    savings_map = {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown"
    }
    employment_map = {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"
    }
    personal_status_map = {
        "A91": "male-div/sep", "A92": "female-married", "A93": "male-single", "A94": "male-married", "A95": "female-single"
    }
    other_debtors_map = {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor"
    }
    property_map = {
        "A121": "real estate", "A122": "life insurance", "A123": "car", "A124": "unknown"
    }
    installment_plan_map = {
        "A141": "bank", "A142": "stores", "A143": "none"
    }
    housing_map = {
        "A151": "rent", "A152": "own", "A153": "for free"
    }
    job_map = {
        "A171": "unskilled-nonresident", "A172": "unskilled-resident", "A173": "skilled", "A174": "management"
    }
    telephone_map = {
        "A191": "none", "A192": "yes"
    }
    foreign_worker_map = {
        "A201": "yes", "A202": "no"
    }

    # replace labels
    df_labeled = df[numeric_features].copy()
    df_labeled['account_status'] = df['account_status'].replace(account_status_map)
    df_labeled['credit_history'] = df['credit_history'].replace(credit_history_map)
    df_labeled['purpose'] = df['purpose'].replace(purpose_map)
    df_labeled['savings'] = df['savings'].replace(savings_map)
    df_labeled['employment_yr'] = df['employment_yr'].replace(employment_map)
    df_labeled['personal_status_sex'] = df['personal_status_sex'].replace(personal_status_map)
    df_labeled['other_debtors'] = df['other_debtors'].replace(other_debtors_map)
    df_labeled['property'] = df['property'].replace(property_map)
    df_labeled['other_installment_plans'] = df['other_installment_plans'].replace(installment_plan_map)
    df_labeled['housing'] = df['housing'].replace(housing_map)
    df_labeled['job'] = df['job'].replace(job_map)
    df_labeled['telephone'] = df['telephone'].replace(telephone_map)
    df_labeled['foreign_worker'] = df['foreign_worker'].replace(foreign_worker_map)

    df_labeled['target'] = df['target'].astype(int)

    return df_labeled