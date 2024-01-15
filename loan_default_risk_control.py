import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import datetime
import feature_engineering_st.preprocessing as preproc
import time
import json
import os

st.title('Loan Default Risk Control')
# DATA_PATH = './data/testA.csv'


# def load_data(nrows):
#     data = pd.read_csv(DATA_PATH, nrows=nrows)
#     return data


with st.sidebar:
    name = st.text_input(label='姓名（中文）', value='张三')
    term = st.selectbox(label='term', options=[3, 5], index=1)
    grade_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade = st.selectbox(label='grade', options=grade_list, index=6)
    dti = st.slider(label='dti', min_value=0.0, max_value=25.0, value=23.0)
    openAcc = st.slider(label='openAcc', min_value=0, max_value=22, value=5)
    # id = st.text_input(label='id', value=10000, placeholder='为贷款清单分配的唯一信用证标识')
    loanAmnt = st.slider(label='loanAmnt', min_value=5000, max_value=50000, value=10000)
    annualIncome = st.slider(label='annualIncome', min_value=30000.0, max_value=150000.0, value=100000.0)
    interestRate = st.slider(label='interestRate', min_value=5.31, max_value=25.0, value=15.0)
    # installment = st.slider(label='installment', min_value=10.0, max_value=1800.0, value=500.0)
    # subGrade_dict = {
    #     'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
    #     'B': ['B1', 'B2', 'B3', 'B4', 'B5'],
    #     'C': ['C1', 'C2', 'C3', 'C4', 'C5'],
    #     'D': ['D1', 'D2', 'D3', 'D4', 'D5'],
    #     'E': ['E1', 'E2', 'E3', 'E4', 'E5'],
    #     'F': ['F1', 'F2', 'F3', 'F4', 'F5'],
    #     'G': ['G1', 'G2', 'G3', 'G4', 'G5']
    # }
    # subGrade = st.selectbox(label='subGrade', options=subGrade_dict[grade])
    employmentTitle = st.slider(label='employmentTitle', min_value=0.0, max_value=400000.0, value=100.0)
    # purpose = st.selectbox(label='purpose', options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # postCode = st.slider(label='postCode', min_value=0.0, max_value=940.0, value=100.0)
    # regionCode = st.slider(label='regionCode', min_value=0, max_value=50, value=40)
    # delinquency_2years = st.slider(label='delinquency_2years', min_value=0, max_value=40, value=20)
    # ficoRangeLow = st.slider(label='ficoRangeLow', min_value=625, max_value=845, value=700)
    ficoRangeHigh = st.slider(label='ficoRangeHigh', min_value=665, max_value=760, value=665)
    totalAcc = st.slider(label='totalAcc', min_value=2, max_value=40, value=30)
    # pubRec = st.slider(label='pubRec', min_value=0, max_value=86, value=5)
    # pubRecBankruptcies = st.slider(label='pubRecBankruptcies', min_value=0, max_value=12, value=1)
    # revolBal = st.slider(label='revolBal', min_value=0.0, max_value=3000000.0, value=1000000.0)
    revolUtil = st.slider(label='revolUtil', min_value=0.0, max_value=900.0, value=100.0)
    # initialListStatus = st.selectbox(label='initialListStatus', options=[0, 1])
    # applicationType = st.selectbox(label='applicationType', options=[0, 1])
    # col1, col2 = st.columns(2)
    # months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # with col1:
    #     selected_month = st.selectbox(label='Month', options=months)
    # with col2:
    #     selected_year = st.slider(label='year', min_value=1944, max_value=2015, value=2015)
    # earliesCreditLine = f'{selected_month}-{selected_year}'
    title = st.slider(label='title', min_value=0.0, max_value=61680.0, value=1000.0)
    # policyCode = st.selectbox(label='policyCode', options=[1.0])
    # n0 = st.sidebar.slider(label='n0', min_value=0, max_value=51, value=10)
    # n1 = st.sidebar.slider(label='n1', min_value=0, max_value=33, value=10)
    n2 = st.sidebar.slider(label='n2', min_value=0, max_value=63, value=5)
    # n3 = st.sidebar.slider(label='n3', min_value=0, max_value=63, value=10)
    # n4 = st.sidebar.slider(label='n4', min_value=0, max_value=49, value=10)
    # n5 = st.sidebar.slider(label='n5', min_value=0, max_value=70, value=10)
    # n6 = st.sidebar.slider(label='n6', min_value=0, max_value=132, value=10)
    # n7 = st.sidebar.slider(label='n7', min_value=0, max_value=79, value=10)
    # n8 = st.sidebar.slider(label='n8', min_value=1, max_value=128, value=10)
    # n9 = st.sidebar.slider(label='n9', min_value=0, max_value=45, value=10)
    # n10 = st.sidebar.slider(label='n10', min_value=0, max_value=82, value=10)
    # n11 = st.sidebar.slider(label='n11', min_value=0, max_value=4, value=2)
    # n12 = st.sidebar.slider(label='n12', min_value=0, max_value=4, value=2)
    # n13 = st.sidebar.slider(label='n13', min_value=0, max_value=39, value=10)
    n14 = st.sidebar.slider(label='n14', min_value=0, max_value=6, value=5)
    employmentLength = st.selectbox(label='employmentLength',
                                    options=['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                                             '6 years', '7 years', '8 years', '9 years', '10+ years'])
    homeOwnership = st.selectbox(label='homeOwnership', options=[0, 1, 2, 3, 4, 5])
    issueDate = st.date_input(label='issueDate', value=datetime.date(2018, 1, 1), min_value=datetime.date(2007, 1, 1),
                              max_value=datetime.date(2019, 1, 1))
    issueDate = issueDate.strftime('%Y-%m-%d')
    verificationStatus = st.selectbox(label='verificationStatus', options=[0, 1, 2], index=2)

# with st.spinner('正在评估，请稍候...'):
# 加载模型和映射字典
clf = load('data/LogisticRegression_best_model.joblib')
woe_maps = load('data/woe_maps.joblib')
bin_boundaries = load('data/bin_boundaries.joblib')
# st.write(bin_boundaries.export()['LoanAmntOannualIncome'])

input_data = {
    'id': [10000],
    'loanAmnt': [loanAmnt],
    'term': [term],
    'interestRate': [interestRate],
    'installment': [500.0],
    'grade': [grade],
    'subGrade': ['A1'],
    'employmentTitle': [employmentTitle],
    'employmentLength': [employmentLength],
    'homeOwnership': [homeOwnership],
    'annualIncome': [annualIncome],
    'verificationStatus': [verificationStatus],
    'issueDate': [issueDate],
    'purpose': [0],
    'postCode': [100.0],
    'regionCode': [40],
    'dti': [dti],
    'delinquency_2years': [20],
    'ficoRangeLow': [700],
    'ficoRangeHigh': [ficoRangeHigh],
    'openAcc': [openAcc],
    'pubRec': [5],
    'pubRecBankruptcies': [1],
    'revolBal': [1000000.0],
    'revolUtil': [revolUtil],
    'totalAcc': [totalAcc],
    'initialListStatus': [0],
    'applicationType': [0],
    'earliesCreditLine': ["Jan-2015'"],
    'title': [title],
    'policyCode': [1.0],
    'n0': [10],
    'n1': [10],
    'n2': [n2],
    'n3': [10],
    'n4': [10],
    'n5': [10],
    'n6': [10],
    'n7': [10],
    'n8': [10],
    'n9': [10],
    'n10': [10],
    'n11': [2],
    'n12': [2],
    'n13': [10],
    'n14': [n14]
}

input_df = pd.DataFrame(data=input_data)

with st.status('正在评估，请稍候...', expanded=True):
    placeholder_1 = st.empty()  # 创建一个空占位符
    placeholder_2 = st.empty()  # 创建一个空占位符
    placeholder_3 = st.empty()  # 创建一个空占位符
    placeholder_1.text('正在评估身份信息...')
    time.sleep(1)
    placeholder_2.text('正在评估财务状况...')  # 更新占位符内容
    time.sleep(1)
    placeholder_3.text('正在评估信用历史...')
    time.sleep(1)

input_df_preprocessed, feature_list = preproc.test_dataset_process(input_df, bin_boundaries)
score_df = input_df_preprocessed.copy()
# st.text('score_df')
# st.write(score_df)

# 应用 WOE 转换
input_df_preprocessed = preproc.apply_woe(input_df_preprocessed, woe_maps, feature_list)
# st.write(input_df_preprocessed)
train_feature_order = ['LoanAmntOannualIncome', 'annualIncome', 'dti', 'dti_grade_interaction', 'ficoRangeHigh',
                       'interestRateOLoanAmnt', 'issueDate_year', 'loanAmnt', 'n14', 'n2', 'openAccOTotalAcc',
                       'revolUtil', 'term', 'title', 'verificationStatus']

# 重新排序测试数据集的列以匹配训练时的顺序
testA_preprocessed = input_df_preprocessed[train_feature_order]
# 进行预测
testA_predictions = clf.predict_proba(testA_preprocessed)[:, 1]
submission = pd.DataFrame({'id': input_df['id'], 'isDefault': testA_predictions})
# st.write(submission)

# 评分卡
# 您需要将 woe_maps 转换成适合的 DataFrame 格式
# 调用 generate_scorecard 函数创建评分卡
# st.write(clf.coef_[0])
# 生成评分卡和偏移量
score_card, offset = preproc.generate_scorecard(clf.coef_[0], train_feature_order, woe_maps)
# st.write(score_card)

scored_sample = preproc.calculate_score_with_card(score_df, score_card, offset)

# st.write(scored_sample)

# 假设你已经有了分数和姓名
score = int(scored_sample.iloc[0])
protected_name = "*" * (len(name) - 1) + name[-1]  # 将除了最后一个字符之外的所有字符替换为星号

with st.spinner("正在生成信用评分，请稍后..."):
    time.sleep(1.5)
    # 显示分数
    st.subheader("我的信用评分", divider=True)
    st.markdown("""
    <style>
    .big-font {
        font-size:35px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<p class="big-font">{score}</p>', unsafe_allow_html=True)

with st.expander(label="了解分数", expanded=True):
    # 使用markdown来格式化和显示评价信息
    st.markdown("**我的分数构成**")
    st.markdown(f"信用评分是对本人（{protected_name}）的综合评分，当前分数 **{score}** 主要由以下信息评估得出：")

    # 身份信息描述
    identity_info = "身份信息描述内容"
    # 财务状况描述
    financial_status = "财务状况描述内容"
    # 信用历史描述
    credit_history = "信用历史描述内容"
    st.markdown(f":gray[身份信息]：&nbsp;&nbsp;&nbsp;&nbsp;{identity_info}")
    st.markdown(f":grey[财务状况]：&nbsp;&nbsp;&nbsp;&nbsp;{financial_status}")
    st.markdown(f":grey[信用历史]：&nbsp;&nbsp;&nbsp;&nbsp;{credit_history}")

st.success('评分完成！', icon="✅")


def determine_approval_status(score):
    if score < 645:
        st.toast("很抱歉，您的申请未通过。", icon="😞")
        return "不通过"
    elif score >= 670:
        st.toast("恭喜！您的申请已通过。", icon="🎉")
        return "通过"
    else:
        st.toast("您的申请需要进一步人工审核。", icon="🔍")
        return "人工审核"


# 创建一个包含所有申请信息的字典
application_details = {
    'term': term,
    'grade': grade,
    'dti': dti,
    'openAcc': openAcc,
    'loanAmnt': loanAmnt,
    'annualIncome': annualIncome,
    'interestRate': interestRate,
    'employmentTitle': employmentTitle,
    'ficoRangeHigh': ficoRangeHigh,
    'totalAcc': totalAcc,
    'revolUtil': revolUtil,
    'title': title,
    'n2': n2,
    'n14': n14,
    'employmentLength': employmentLength,
    'homeOwnership': homeOwnership,
    'issueDate': issueDate,
    'verificationStatus': verificationStatus
}
# 将字典转换为字符串以便保存
application_details_str = json.dumps(application_details)
# 评分和审批状态
approval_status = determine_approval_status(score)
# 创建一个新的DataFrame来保存结果及申请信息
results_df = pd.DataFrame({
    '客户姓名': [name],
    '信用评分': [score],
    '自动审批状态': [approval_status],
    '申请细节': [application_details_str]
})
# 保存到CSV文件
results_path = 'data/approval_results.csv'
results_df.to_csv(results_path, index=False, mode='a', header=not os.path.exists(results_path))
