# Load-Default-Risk-Control

## Introduction
"Load-Default-Risk-Control" is a sophisticated Streamlit web application focused on loan default risk management. This app features an advanced logistic regression scorecard model for the automated evaluation of loan default risks, enabling efficient, data-driven decisions in financial risk assessment.

## Key Features
- **Automated Risk Score Assessment**: Utilizes a logistic regression scorecard for quick and accurate risk evaluations.
- **Threshold-Based Automated Decision Making**: Approves or rejects loan applications based on set risk score thresholds, optimizing decision-making speed and accuracy.
- **Manual Review Indication**: Flags applications that need further human evaluation, ensuring comprehensive risk assessment.
- **Interactive and User-Friendly Interface**: Inputs are easily managed via a sidebar, allowing for streamlined data entry and interaction.
- **Real-Time Processing and Analysis**: Capable of processing input data in real-time for immediate risk scoring and decision-making.

## How to Use
1. **Access the App**: Go to [Load-Default-Risk-Control](https://loan-default-risk-control-st.streamlit.app/) on the Streamlit Community Cloud.
2. **Input Data via Sidebar**: Use the sidebar to input loan-related information such as loan amount, term, grade, annual income, and other financial details.
3. **Submit for Risk Assessment**: After data input, the app processes the information using the scorecard model and computes a risk score.
4. **Automated Loan Decision**:
   - Applications exceeding the risk score threshold are automatically approved.
   - Applications below the threshold are automatically rejected.
   - Applications close to the threshold are flagged for manual review.
5. **View Assessment Summaries**: Navigate to a separate page within the app to view all assessment summaries and comprehensive results.

## Technical Details
- **Built With**: Developed using Python, featuring libraries like Pandas, NumPy, scikit-learn for data handling, machine learning operations, and Plotly for interactive visualizations.
- **Scorecard Model**: Integrates logistic regression for calculating risk scores, which are essential for automated decision-making.
- **Dynamic Data Processing**: Implements efficient feature engineering and real-time data processing for immediate risk analysis.
