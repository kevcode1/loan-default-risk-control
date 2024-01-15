import pandas as pd
import numpy as np
import math
from joblib import dump, load


def convert_employment_length(el):
    if el == '< 1 year':
        return 0
    elif el == '10+ years':
        return 10
    elif pd.isna(el):
        return np.nan
    else:
        return int(el.split(' ')[0])


def vars_encode(df):
    # 低维类别变量 ont-hot
    df = pd.get_dummies(df,
                        columns=['grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose',
                                 'regionCode',
                                 'initialListStatus'], dummy_na=True, dtype=int)
    # 高维类别变量
    # 直接对这些特征应用独热编码会导致数据维度过大，从而增加模型训练的复杂性和计算成本。
    # 转换为基于计数和排名的数值特征。
    for var in ['employmentTitle', 'postCode', 'title']:
        # 替换缺失值
        df[var].fillna(-1, inplace=True)  # 假设 -1 在数据中是唯一的
        df[var + '_counts'] = df.groupby([var])['id'].transform('count')
        df[var + '_rank'] = df.groupby([var])['id'].rank(ascending=False).astype(int)
        # 将特殊标记的值再次转换为 NaN
        df[var].replace(-1, np.nan, inplace=True)

    df.drop(['employmentTitle', 'postCode', 'title'], axis=1, inplace=True)

    return df


def subgrade_trans(x, grade_dict):
    # 字母部分转换为数字
    grade_part = grade_dict[x[0]]
    # 数字部分转换为整数
    num_part = int(x[1])

    return grade_part * 10 + num_part


# 假设 dataset 是完整的数据集，已经包含了目标变量 'isDefault' 和所需特征
# feature_list 是需要计算 WOE 的特征列表
# 定义 WOE 计算函数
def cal_woe(df, feature_list, target):
    # 创建一个空的 DataFrame 用于存储 WOE 和 IV
    woe_iv_df = pd.DataFrame(columns=['feature', 'value', 'woe', 'iv'])
    # 对于每个特征进行 WOE 和 IV 的计算
    for feature in feature_list:
        # 对特征进行分组，并计算每组的坏客户数和总客户数
        woe_df = df.groupby(feature)[target].agg(bad='sum', total='count')
        # 计算好客户数
        woe_df['good'] = woe_df['total'] - woe_df['bad']
        # 避免除以0，将0值替换为极小的正数
        woe_df['bad'] = woe_df['bad'].replace(0, 0.0001)
        woe_df['good'] = woe_df['good'].replace(0, 0.0001)
        # 计算每组的坏客户比率和好客户比率
        woe_df['bad_rate'] = woe_df['bad'] / df[target].sum()
        woe_df['good_rate'] = woe_df['good'] / (df[target].count() - df[target].sum())
        # 计算 WOE 值
        woe_df['woe'] = np.log(woe_df['bad_rate'] / woe_df['good_rate'])
        # 计算每组的 IV 值
        woe_df['iv'] = (woe_df['bad_rate'] - woe_df['good_rate']) * woe_df['woe']
        # 重置索引以获取特征值
        woe_df.reset_index(inplace=True)
        # 汇总特征的 WOE 和 IV
        woe_df['feature'] = feature
        woe_df.rename(columns={feature: 'value'}, inplace=True)
        # 选择需要的列
        woe_iv_df = pd.concat([woe_iv_df, woe_df[['feature', 'value', 'woe', 'iv']]])
    # 计算每个特征的总 IV
    iv_df = woe_iv_df.groupby('feature')['iv'].sum().reset_index().rename(columns={'iv': 'total_iv'})
    woe_iv_df = woe_iv_df.merge(iv_df, on='feature')

    return woe_iv_df


def map_to_nearest_bin(value, bin_tuple):
    bins, labels = bin_tuple  # 提取边界和标签
    if value <= bins[0]:
        return bins[0]  # 如果值小于最小边界，映射到最小边界值
    elif value > bins[-1]:
        return bins[-1]  # 如果值大于最大边界，映射到最大边界值
    else:
        return value  # 如果值在边界内，直接返回该值


def apply_woe(df, woe_maps, feature_list):
    for feature in feature_list:
        if feature in woe_maps:
            df[feature] = df[feature].apply(lambda x: woe_maps[feature].get(x, x))
    return df


def test_dataset_process(dataset, bin_boundaries):
    # 债务负值异常值处理
    dataset.loc[dataset['dti'] < 0, ['dti']] = 0
    # 1.2 数据清洗与预处理
    single_value_columns = ['policyCode']
    # print(f'There are {len(single_value_columns)} columns in train_csv with single value.')
    dataset.drop(single_value_columns, axis=1, inplace=True)

    imbalance_list = ['applicationType', 'n11', 'n12']
    # print(f'变量{imbalance_list}分布不均匀，需要删除')
    dataset.drop(imbalance_list, axis=1, inplace=True)

    # employmentLength进行转换到数值
    dataset['employmentLength'] = dataset['employmentLength'].apply(convert_employment_length)
    # 处理 issueDate
    # earliesCreditLine进行预处理
    dataset['issueDate_year'] = dataset['issueDate'].apply(lambda x: int(x.split('-')[0]))
    dataset['issueDate_month'] = dataset['issueDate'].apply(lambda x: int(x.split('-')[1]))
    dataset['issueDate'] = pd.to_datetime(dataset['issueDate'])
    dataset['earliesCreditLine'] = pd.to_datetime(
        dataset['earliesCreditLine'].apply(lambda x: x[-4:] + '-' + x[:3] + '-01'))
    dataset['credit_history_days'] = (dataset['issueDate'] - dataset['earliesCreditLine']).dt.days
    dataset['credit_history_days'] = dataset['credit_history_days'].astype(int)
    dataset.drop(['issueDate', 'earliesCreditLine'], axis=1, inplace=True)

    # 处理grade
    grade_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    dataset['grade'] = dataset['grade'].apply(lambda x: grade_dict[x])
    # 处理subgrade
    dataset['subGrade'] = dataset['subGrade'].apply(lambda x: subgrade_trans(x, grade_dict))
    dataset['ficoRangeDiff'] = dataset['ficoRangeHigh'] - dataset['ficoRangeLow']
    # 大致年还款额
    dataset['annual_loan_payment'] = dataset['loanAmnt'] / dataset['term']
    # 年收入与年还款额的比值，反映了还款压力
    dataset['income_to_payment_ratio'] = dataset['annualIncome'] / dataset['annual_loan_payment']
    # 选择所有匿名特征
    anonymous_features = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n13',
                          'n14']  # 计算所有匿名特征的均值
    dataset['mean_anonymous'] = dataset[anonymous_features].mean(axis=1)

    # 利率/贷款总额
    dataset['interestRateOLoanAmnt'] = dataset['interestRate'] / dataset['loanAmnt']
    # 将年收入为零的替换为一个小的正数或平均值
    dataset['annualIncome'] = dataset['annualIncome'].replace(0, dataset['annualIncome'].mean())
    # 年收入/贷款总额
    dataset['LoanAmntOannualIncome'] = dataset['loanAmnt'] / dataset['annualIncome']
    # 年收入*就业年限
    dataset['income_employment_interaction'] = dataset['annualIncome'] * dataset['employmentLength']
    # 未结信用额度的数量/当前的信用额度总数，代表借款人的信誉度
    dataset['openAccOTotalAcc'] = dataset['openAcc'] / dataset['totalAcc']
    # 未结信用额度的数量/最早信用额度开立距今时间
    dataset['openAccOcredit_history_days'] = dataset['openAcc'] / dataset['credit_history_days']
    # 相对于信用分数范围的信贷周转余额的使用率
    dataset['revolBalOficoRangeDiff'] = dataset['revolBal'] / dataset['ficoRangeDiff']
    # 信贷周转余额与年收入的比例
    dataset['revolBalOannualIncome'] = dataset['revolBal'] / dataset['annualIncome']
    # 分期付款金额与月收入的比例
    dataset['installmentOmonthIncome'] = dataset['installment'] / (dataset['annualIncome'] / 12)
    # 平均账户年龄
    dataset['average_account_age'] = dataset['credit_history_days'] / dataset['totalAcc']
    # 公共记录数量与总账户数的比例
    dataset['pub_records_ratio'] = dataset['pubRec'] / dataset['totalAcc']
    # 破产记录数量与总账户数的比例
    dataset['bankruptcies_ratio'] = dataset['pubRecBankruptcies'] / dataset['totalAcc']
    # 债务收入比与贷款等级的交互作用
    dataset['dti_grade_interaction'] = dataset['dti'] * dataset['grade']
    # 逾期与总账户比例
    dataset['delinq_to_total_acc_ratio'] = dataset['delinquency_2years'] / dataset['totalAcc']

    # print(f'缺失值处理之前，{dataset.isnull().sum()}')

    # 缺失值处理
    dataset.fillna(-9999, inplace=True)

    # print(f'缺失值处理之后，{dataset.isnull().sum()}')

    features = [f for f in dataset.columns.tolist() if f not in ['id', 'isDefault']]
    dataset = dataset[features]

    features_to_bin = ['LoanAmntOannualIncome', 'annualIncome', 'dti', 'dti_grade_interaction', 'ficoRangeHigh',
                       'interestRateOLoanAmnt', 'issueDate_year', 'loanAmnt', 'n14', 'n2', 'openAccOTotalAcc',
                       'revolUtil', 'term', 'title', 'verificationStatus']

    # 映射到最近的分箱边界
    dataset = bin_boundaries.transform(dataset, labels=False)
    dataset = dataset[features_to_bin]

    # print(f'cut之后，{dataset.isnull().sum()}')

    return dataset, features_to_bin


def generate_scorecard(model_coef, feature_names, woe_maps, pdo=20, base_score=600):
    factor = pdo / np.log(2)
    odds = 1 / 20  # 假设的赔率
    offset = base_score - factor * np.log(odds)
    score_rows = []  # 存储所有评分卡行的列表

    for i, feature in enumerate(feature_names):
        coef = model_coef[i]
        woe_dict = woe_maps[feature]
        for binning, woe_value in woe_dict.items():
            score = round(-factor * coef * woe_value)
            score_rows.append({'Feature': feature, 'Binning': binning, 'Score': score})

    scorecard = pd.DataFrame(score_rows)
    return scorecard, offset


def map_to_score(row, score_card):
    score = 0
    for _, card_row in score_card.iterrows():
        feature = card_row['Feature']
        if pd.isna(row[feature]):
            continue  # 如果数据中的特征值是 NaN，则跳过
        if row[feature] == card_row['Binning']:
            score += card_row['Score']
    return score


def calculate_score_with_card(df, score_card, base_score):
    scores = df.apply(lambda row: map_to_score(row, score_card), axis=1)
    total_scores = scores + base_score
    return total_scores.astype(int)
