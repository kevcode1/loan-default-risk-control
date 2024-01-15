import streamlit as st
import pandas as pd
import plotly.express as px
import json

# 加载审批结果数据
approval_results = pd.read_csv('data/approval_results.csv')


# 定义一个函数，用于从 JSON 字符串中提取 issueDate
def extract_issue_date(json_str):
    try:
        details = json.loads(json_str.replace("'", '"'))  # 替换单引号为双引号
        return details.get('issueDate', None)
    except json.JSONDecodeError:
        return None


# 应用这个函数到申请细节列
approval_results['issueDate'] = approval_results['申请细节'].apply(extract_issue_date)

# 将 issueDate 转换为 datetime 对象并提取年份
approval_results['issueDate'] = pd.to_datetime(approval_results['issueDate'])
approval_results['Year'] = approval_results['issueDate'].dt.year
# 对每年的每种审批状态进行计数
yearly_status_counts = approval_results.groupby(['Year', '自动审批状态']).size().unstack(fill_value=0)
# st.write(yearly_status_counts.index)
# st.write(yearly_status_counts.columns)
# 创建折线图
fig = px.line(yearly_status_counts, x=yearly_status_counts.index, y=yearly_status_counts.columns,
              labels={'value': '数量', 'variable': '自动审批状态'}, title='自动审批状态随时间的变化')
# 显示图表
st.plotly_chart(fig)

st.subheader("自动评估结果", divider=True)

# 设置每页显示的记录数
records_per_page = 5
# 初始化session_state
if 'page' not in st.session_state:
    st.session_state['page'] = 0
# 计算总页数
total_pages = len(approval_results) // records_per_page + (1 if len(approval_results) % records_per_page > 0 else 0)
# 显示当前页的记录
start = st.session_state['page'] * records_per_page
end = start + records_per_page
# 分页导航
prev, _, next = st.columns([1, 6, 1])
if prev.button("上一页"):
    if st.session_state['page'] > 0:
        st.session_state['page'] -= 1
if next.button("下一页"):
    if st.session_state['page'] < total_pages - 1:
        st.session_state['page'] += 1
# 在页面上显示页码和总评估数
status_col, page_col = st.columns([1, 1])  # 调整列的宽度比例
with status_col:
    total_evaluations = len(approval_results)
    st.write(f"总评估数: {total_evaluations}")
with page_col:
    st.write(f"Page: {st.session_state['page'] + 1} of {total_pages}")
# 对当前页的审批结果创建一个可展开的视图
for index, row in approval_results.iloc[start:end].iterrows():
    # 将 datetime 对象格式化为日期字符串
    row['issueDate'] = row['issueDate'].strftime('%Y-%m-%d') if pd.notnull(row['issueDate']) else None
    # 创建一个容器来整合每条记录的信息
    with st.container(border=True):
        # 使用列来并排显示数据和展开器
        col1, col2 = st.columns(spec=[0.5, 0.5])
        # 在第一列中显示审批结果的基本信息
        with col1:
            st.write(pd.DataFrame(row).T.drop(columns=['申请细节', 'issueDate', 'Year']))
        # 在第二列中使用expander显示申请细节
        with col2:
            with st.expander("查看详情"):
                application_details = json.loads(row['申请细节'])
                st.write(pd.DataFrame([application_details]))
